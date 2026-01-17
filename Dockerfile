FROM condaforge/miniforge3:24.11.0-0

WORKDIR /app

# Faster, cleaner conda
RUN conda config --set channel_priority strict

# System deps (optional but useful)
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy only requirements first to leverage Docker cache
COPY requirements.txt /app/requirements.txt

# Create env and install deps
# Important: pin numpy<2 because faiss wheels/libs often not compatible with numpy 2.x
RUN conda install -y -c conda-forge python=3.11 faiss-cpu && \
    pip install --no-cache-dir -r requirements.txt

# Copy project code
COPY . /app

# Default command: run telegram bot
CMD ["python", "bot/bot.py"]