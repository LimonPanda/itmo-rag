import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
from mistralai import Mistral
import re
import time

import faiss
from aiogram import Bot, Dispatcher, F
from aiogram.filters import CommandStart
from aiogram.types import Message, InlineKeyboardMarkup, InlineKeyboardButton, CallbackQuery
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

import os


logging.basicConfig(level=logging.INFO)

DATA_JSONL = Path("data/processed/recipes.jsonl")
INDEX_DIR = Path("data/index")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

TOPK = 25
TOPN = 5


@dataclass
class Candidate:
    recipe_id: str
    title: str
    url: str
    matched_ingredients: List[str]
    score: float


def load_doc_ids() -> List[str]:
    with open(INDEX_DIR / "doc_ids.json", "r", encoding="utf-8") as f:
        return json.load(f)


def load_recipes_map() -> Dict[str, Dict]:
    m: Dict[str, Dict] = {}
    with open(DATA_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            d = json.loads(line)
            rid = str(d.get("id", ""))
            if rid:
                m[rid] = d
    return m


def encode_query(model: SentenceTransformer, query: str):
    vec = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    return vec.astype("float32")


def tokenize_ingredients(s: str) -> List[str]:
    parts = [p.strip().lower() for p in s.split(",")]
    return [p for p in parts if p]


def safe_json_loads(text: str) -> Dict:
    """
    Tries to parse JSON even if model returned extra text.
    Extracts first {...} block.
    """
    text = (text or "").strip()
    if not text:
        raise ValueError("Empty LLM response")

    # try direct
    try:
        return json.loads(text)
    except Exception:
        pass

    # try extract {...}
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(m.group(0))


def parse_user_input_fallback(text: str) -> Dict:
    """
    Robust fallback without LLM:
    - Extracts dish_type from keywords (RU/EN)
    - Extracts ingredients from free text (RU/EN), ignores non-food
    - Returns EN-only fields (via _normalize_parsed)
    """
    t = (text or "").lower()

    # dish type from keywords (extend as needed)
    dish_map = {
        "суп": "soup",
        "salad": "salad",
        "салат": "salad",
        "паста": "pasta",
        "pasta": "pasta",
        "десерт": "dessert",
        "dessert": "dessert",
        "рагу": "stew",
        "stew": "stew",
    }
    dish_type = ""
    for k, v in dish_map.items():
        if k in t:
            dish_type = v
            break

    # simple preferences
    pref_map = {
        "острый": "spicy",
        "острое": "spicy",
        "spicy": "spicy",
        "быстро": "quick",
        "quick": "quick",
        "вегетариан": "vegetarian",
        "vegetarian": "vegetarian",
        "сливоч": "creamy",
        "creamy": "creamy",
        "томат": "tomato",
        "tomato": "tomato",
    }
    preferences = []
    for k, v in pref_map.items():
        if k in t and v not in preferences:
            preferences.append(v)

    # tokenize words (keep letters only)
    words = re.findall(r"[a-zа-яё\-]+", t)

    # obvious non-food junk (extend if needed)
    non_food = {
        "пенопласт", "гантеля", "гантели", "гвозди", "мыло", "шампунь",
        "лампочка", "батарейка", "пакет", "салфетки"
    }

    # pass tokens through _normalize_parsed which includes ru->en mapping & filters
    ingredients = [w for w in words if w not in non_food]

    d = {"dish_type": dish_type, "preferences": preferences, "ingredients": ingredients}
    return _normalize_parsed(d)


def _normalize_parsed(data: Dict) -> Dict:
    dish_type = str(data.get("dish_type", "")).strip().lower()
    prefs = data.get("preferences", [])
    ings = data.get("ingredients", [])

    if not isinstance(prefs, list):
        prefs = []
    if not isinstance(ings, list):
        ings = []

    prefs = [str(x).strip().lower() for x in prefs if str(x).strip()]
    ings = [str(x).strip().lower() for x in ings if str(x).strip()]

    # Minimal RU->EN mapping to enforce English output even if model slips.
    ru_en = {
        # dish types (ru -> en)
        "суп": "soup",
        "салат": "salad",
        "паста": "pasta",
        "десерт": "dessert",
        "завтрак": "breakfast",
        "ужин": "dinner",
        "обед": "lunch",
        "рагу": "stew",
        "тушеное": "stew",

        # preferences (ru -> en)
        "острый": "spicy",
        "острое": "spicy",
        "быстро": "quick",
        "быстрый": "quick",
        "легкое": "light",
        "легкий": "light",
        "сытное": "hearty",
        "вегетарианское": "vegetarian",
        "вегетарианский": "vegetarian",
        "без мяса": "meatless",
        "сливочное": "creamy",
        "томатное": "tomato",

        # ingredients (ru -> en) - common ones for your dataset
        "курица": "chicken",
        "куриное": "chicken",
        "куриный": "chicken",
        "лук": "onion",
        "репчатый лук": "onion",
        "морковь": "carrot",
        "картофель": "potato",
        "картошка": "potato",
        "чеснок": "garlic",
        "помидор": "tomato",
        "помидоры": "tomatoes",
        "томат": "tomato",
        "огурец": "cucumber",
        "перец": "pepper",
        "болгарский перец": "bell pepper",
        "грибы": "mushrooms",
        "шампиньоны": "mushrooms",
        "сыр": "cheese",
        "молоко": "milk",
        "сливки": "cream",
        "сметана": "sour cream",
        "масло": "butter",
        "яйца": "eggs",
        "яйцо": "egg",
        "рис": "rice",
        "макароны": "pasta",
        "мука": "flour",
        "сахар": "sugar",
        "соль": "salt",
    }

    def map_ru_to_en(s: str) -> str:
        s = s.strip().lower()
        # direct match
        if s in ru_en:
            return ru_en[s]
        # try simple normalization for common multiword like "без мяса"
        if "без мяса" in s:
            return "meatless"
        return s

    # Enforce dish_type in English if possible
    dish_type = map_ru_to_en(dish_type)

    # Heuristic "looks like English": allow ascii letters/spaces/hyphen only.
    def looks_english(t: str) -> bool:
        for ch in t:
            if ch.isascii():
                continue
            return False
        return True

    # Filter out receipt noise + numbers/currency
    def ok_token(t: str) -> bool:
        if len(t) < 2:
            return False
        if any(ch.isdigit() for ch in t):
            return False
        bad = {"руб", "р", "₽", "$", "eur", "€", "шт", "кг", "г", "ml", "l"}
        if t in bad:
            return False
        return True

    # Apply RU->EN mapping for preferences/ingredients (in case model returns RU)
    prefs = [map_ru_to_en(p) for p in prefs]
    ings = [map_ru_to_en(i) for i in ings]

    # Post-filter: keep only clean tokens
    prefs = [p for p in prefs if ok_token(p)]
    ings = [i for i in ings if ok_token(i)]

    # If still not English, try to drop non-ascii (we prefer empty over wrong language)
    prefs = [p for p in prefs if looks_english(p)]
    ings = [i for i in ings if looks_english(i)]

    return {
        "dish_type": dish_type if looks_english(dish_type) else "",
        "preferences": prefs[:8],
        "ingredients": ings[:25],
    }


def parse_user_input_llm_mistral(client: Mistral, model: str, text: str) -> Dict:
    """
    Extract dish_type, preferences, ingredients from arbitrary user text (any language).
    ALWAYS return English-only JSON fields (dish_type/prefs/ingredients).
    """
    system = (
    "You are a strict JSON information-extraction parser for a cooking assistant.\n"
    "Input can be in ANY language (including Russian) and may contain irrelevant items.\n\n"
    "Return ONLY a valid JSON object (no markdown, no extra text).\n"
    "Schema:\n"
    "{"
    "\"dish_type\": string, "
    "\"preferences\": array[string], "
    "\"ingredients\": array[string]"
    "}\n\n"
    "Rules:\n"
    "- Output MUST be in ENGLISH only.\n"
    "- dish_type: infer from text if possible (soup/salad/pasta/dessert/stew), else empty.\n"
    "- preferences: short English tags (spicy, quick, vegetarian, creamy, tomato, etc.).\n"
    "- ingredients: ONLY edible food ingredients. Ignore non-food items.\n"
    "- If user lists things like 'styrofoam', 'dumbbell', 'tool', ignore them.\n"
    "- Do not include quantities, brands, prices.\n"
    "- Max ingredients: 25.\n\n"
    "Examples:\n"
    "Input (RU): 'Хочу острый суп, у меня есть курица, пенопласт, лук, гантеля и картошка'\n"
    "Output: {\"dish_type\":\"soup\",\"preferences\":[\"spicy\"],\"ingredients\":[\"chicken\",\"onion\",\"potato\"]}\n"
    )

    user = f"USER_TEXT:\n{text}"

    last_err: Exception | None = None
    for attempt in range(3):
        try:
            resp = client.chat.complete(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
            )
            raw = (resp.choices[0].message.content or "").strip()
            data = safe_json_loads(raw)
            return _normalize_parsed(data)
        except Exception as e:
            last_err = e
            # small backoff; helps with rate/capacity bursts
            time.sleep(0.6 * (attempt + 1))

    raise last_err if last_err else RuntimeError("LLM parse failed")


def retrieve_and_rerank(
    index: faiss.Index,
    doc_ids: List[str],
    recipes: Dict[str, Dict],
    model: SentenceTransformer,
    dish_type: str,
    ingredients: str,
    topk: int = TOPK,
    topn: int = TOPN,
) -> List[Candidate]:
    dish_kw = dish_type.strip().lower()
    user_ing = tokenize_ingredients(ingredients)
    user_set = set(user_ing)

    query = dish_type.strip()
    if ingredients.strip():
        query = (query + " " + ingredients.strip()).strip()

    qvec = encode_query(model, query)
    sims, idxs = index.search(qvec, topk)

    candidates: List[Tuple[str, float]] = []
    for j in range(idxs.shape[1]):
        i = int(idxs[0, j])
        if 0 <= i < len(doc_ids):
            candidates.append((doc_ids[i], float(sims[0, j])))

    alpha = 0.35
    beta = 0.15

    out: List[Candidate] = []
    for rid, sim in candidates:
        r = recipes.get(rid)
        if not r:
            continue
        title = str(r.get("title", ""))
        title_l = title.lower()
        url = str(r.get("source_url", ""))

        ingr_norm = [str(x).lower() for x in r.get("ingredients_normalized", [])]
        ingr_set = set(ingr_norm)

        overlap = len(user_set & ingr_set) if user_set else 0
        overlap_ratio = (overlap / max(1, len(user_set))) if user_set else 0.0

        # Hard filter: if user provided ingredients, require at least 1 overlap
        if user_set and overlap == 0:
            continue

        score = float(sim) + alpha * overlap_ratio
        if dish_kw and dish_kw in title_l:
            score += beta

        out.append(
            Candidate(
                recipe_id=rid,
                title=title,
                url=url,
                matched_ingredients=sorted(list(user_set & ingr_set))[:10],
                score=score,
            )
        )

    out.sort(key=lambda x: x.score, reverse=True)
    return out[:topn]


def format_full_recipe(recipe: Dict, matched: List[str]) -> str:
    title = str(recipe.get("title", "")).strip()
    url = str(recipe.get("source_url", "")).strip()

    ingredients_raw = recipe.get("ingredients_raw", []) or []
    instructions = str(recipe.get("instructions", "")).strip()

    matched_text = ", ".join(matched) if matched else "—"
    ingr_text = "\n".join([f"- {x}" for x in ingredients_raw]) if ingredients_raw else "—"

    # Telegram limit is ~4096 chars; keep it safe
    max_len = 3800
    body = (
        f"{title}\n\n"
        f"Matched ingredients:\n{matched_text}\n\n"
        f"Ingredients:\n{ingr_text}\n\n"
        f"Instructions:\n{instructions}\n\n"
        f"Link: {url}"
    )
    if len(body) > max_len:
        # Truncate instructions first
        head = (
            f"{title}\n\n"
            f"Matched ingredients:\n{matched_text}\n\n"
            f"Ingredients:\n{ingr_text}\n\n"
            f"Instructions:\n"
        )
        tail = f"\n\nСсылка: {url}"
        available = max_len - len(head) - len(tail) - 20
        instructions_cut = (instructions[:available] + "...") if available > 0 else "..."
        body = head + instructions_cut + tail

    return body


def format_recipe_part1(recipe: Dict, matched: List[str]) -> str:
    title = str(recipe.get("title", "")).strip()
    ingredients_raw = recipe.get("ingredients_raw", []) or []

    matched_text = ", ".join(matched) if matched else "—"
    ingr_text = "\n".join([f"- {x}" for x in ingredients_raw]) if ingredients_raw else "—"

    return (
        f"{title}\n\n"
        f"Matched ingredients:\n{matched_text}\n\n"
        f"Ingredients:\n{ingr_text}"
    )


def format_recipe_part2(recipe: Dict) -> str:
    instructions = str(recipe.get("instructions", "")).strip()
    url = str(recipe.get("source_url", "")).strip()

    if not instructions:
        instructions = "—"

    # Keep message within Telegram limit
    max_len = 3800
    if len(instructions) > max_len:
        instructions = instructions[:max_len] + "..."

    return (
        f"Instructions:\n{instructions}\n\n"
        f"Link: {url}"
    )


def build_choose_level_keyboard(cands: List[Candidate]) -> InlineKeyboardMarkup:
    buttons = []
    for i, c in enumerate(cands, start=1):
        buttons.append(
            [InlineKeyboardButton(text=f"{i}) {c.title}", callback_data=f"pick:{c.recipe_id}")]
        )
    # extra button at level 2
    buttons.append([InlineKeyboardButton(text="Make another query", callback_data="action:new_query")])
    return InlineKeyboardMarkup(inline_keyboard=buttons)


def build_recipe_level_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        inline_keyboard=[
            [InlineKeyboardButton(text="Back to recipe selection", callback_data="action:back_to_list")],
            [InlineKeyboardButton(text="Make another query", callback_data="action:new_query")],
        ]
    )


# def build_candidates_keyboard(cands: List[Candidate]) -> InlineKeyboardMarkup:
#     buttons = []
#     for i, c in enumerate(cands, start=1):
#         buttons.append(
#             [InlineKeyboardButton(text=f"{i}) {c.title}", callback_data=f"pick:{c.recipe_id}")]
#         )
#     return InlineKeyboardMarkup(inline_keyboard=buttons)


async def main() -> None:
    load_dotenv()
    mistral_key = os.getenv("MISTRAL_API_KEY", "").strip()
    mistral_model = os.getenv("MISTRAL_MODEL", "mistral-small-latest").strip()
    mistral_client = Mistral(api_key=mistral_key) if mistral_key else None
    token = os.getenv("BOT_TOKEN")
    if not token:
        raise RuntimeError("BOT_TOKEN is not set. Put it into .env")

    if not (INDEX_DIR / "faiss.index").exists():
        raise RuntimeError("FAISS index not found. Build it first: python rag/index.py ...")

    bot = Bot(token=token)
    dp = Dispatcher()

    index = faiss.read_index(str(INDEX_DIR / "faiss.index"))
    doc_ids = load_doc_ids()
    recipes = load_recipes_map()
    model = SentenceTransformer(MODEL_NAME)

    user_state: Dict[int, Dict] = {}
    # structure:
    # user_state[user_id] = {
    #   "candidates": {recipe_id: Candidate, ...},
    #   "candidates_order": [recipe_id1, recipe_id2, ...],
    #   "last_parsed_echo": "строка для показа (опционально)"
    # }

    @dp.message(CommandStart())
    async def start(m: Message):
        text = (
            "Hello! I will find recipes based on your request.\n\n"
            "Send me what you want to cook and what ingridients do you have at hand\n\n"

            "Example:\n"
            "I want spicy soup and i have chicken, carrots and onion\n"
            "or just\n"
            "Pasta tomatoes, mushrooms"
        )
        await m.answer(text)

    @dp.message(F.text)
    async def handle_query(m: Message):
        text = m.text or ""
        logging.info("User text: %s", text)
        logging.info("Mistral enabled: %s", bool(mistral_client))
        try:
            if mistral_client:
                parsed = parse_user_input_llm_mistral(mistral_client, mistral_model, text)
                logging.info("LLM parsed: %s", parsed)
            else:
                parsed = parse_user_input_fallback(text)
                logging.info("Fallback parsed: %s", parsed)
        except Exception as e:
            logging.exception("Parsing failed, fallback used: %s", e)
            parsed = parse_user_input_fallback(text)
            await m.answer(
                "I can't call my smart friend LLM right now, he appears to be busy "
                "I will try to process the request without it. "
                "If you don't like the result —  try sending ingredients as a list separated by commas."
            )

        dish_type = parsed.get("dish_type", "")
        preferences = parsed.get("preferences", [])
        ingredients_list = parsed.get("ingredients", [])

        if not dish_type and not ingredients_list:
            await m.answer(
                "Could not understand the request. Examples:\n"
                "- «I want spicy soup, I have chicken and carrots»\n"
                "- or paste receipt text and specify what you are looking for (e.g.: «I want creamy pasta»)\n"
            )
            return

        ingredients = ", ".join(ingredients_list)
        dish_plus_prefs = " ".join([dish_type] + preferences).strip()

        cands = retrieve_and_rerank(
            index=index,
            doc_ids=doc_ids,
            recipes=recipes,
            model=model,
            dish_type=dish_plus_prefs,
            ingredients=ingredients,
        )

        if not cands:
            await m.answer("No recipes found for your query. Try to specify ingredients or preferences.")
            return

        uid = m.from_user.id

        parsed_echo = (
            "Understood request as:\n"
            f"- dish_type: {dish_type or '—'}\n"
            f"- preferences: {', '.join(preferences) if preferences else '—'}\n"
            f"- ingredients: {', '.join(ingredients_list) if ingredients_list else '—'}\n"
        )
        user_state[uid] = {
            "candidates": {c.recipe_id: c for c in cands},
            "candidates_order": [c.recipe_id for c in cands],
            "last_parsed_echo": parsed_echo,
        }

        await m.answer(
            parsed_echo + "\nFound options. Choose a recipe:",
            reply_markup=build_choose_level_keyboard(cands),
        )

    @dp.callback_query(F.data.startswith("pick:"))
    async def pick(cb: CallbackQuery):
        rid = cb.data.split(":", 1)[1]
        uid = cb.from_user.id

        st = user_state.get(uid, {})
        cand_map = st.get("candidates", {})
        cand = cand_map.get(rid)

        if not cand:
            await cb.message.answer("Selected recipe not found. Please make a new request.")
            await cb.answer()
            return

        full = recipes.get(rid)
        if not full:
            await cb.message.answer("Could not load the full recipe. Try selecting another one.")
            await cb.answer()
            return

        msg1 = format_recipe_part1(full, cand.matched_ingredients)
        msg2 = format_recipe_part2(full)

        await cb.message.answer(msg1)
        await cb.message.answer(msg2, reply_markup=build_recipe_level_keyboard())
        await cb.answer()
    @dp.callback_query(F.data == "action:back_to_list")
    async def back_to_list(cb: CallbackQuery):
        uid = cb.from_user.id
        st = user_state.get(uid, {})
        cand_map = st.get("candidates", {})
        order = st.get("candidates_order", [])
        if not cand_map or not order:
            await cb.message.answer("Recipe list not found. Make a new request.")
            await cb.answer()
            return

        cands = [cand_map[rid] for rid in order if rid in cand_map]
        parsed_echo = st.get("last_parsed_echo", "")

        await cb.message.answer(
            (parsed_echo + "\n") if parsed_echo else "" + "Choose a recipe:",
            reply_markup=build_choose_level_keyboard(cands),
        )
        await cb.answer()


    @dp.callback_query(F.data == "action:new_query")
    async def new_query(cb: CallbackQuery):
        # сброс состояния
        uid = cb.from_user.id
        user_state.pop(uid, None)

        await cb.message.answer(
            "Ok. Send a new request in any format.\n"
            "Examples:\n"
            "- «I want spicy soup, I have chicken and carrots»\n"
            "- or 2 lines:\nsoup\nchicken, carrots, onion"
        )
        await cb.answer()


    await dp.start_polling(bot)


if __name__ == "__main__":
    asyncio.run(main())
