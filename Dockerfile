FROM python:3.11-slim

# Install system dependencies needed for Pillow and build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libjpeg-dev \
    zlib1g-dev \
    libpng-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first for efficient caching
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --no-cache-dir -r /app/requirements.txt

# Copy application code
COPY . /app

# Expose port (default 4000)
ENV PORT=4000
EXPOSE 4000

# Default command
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "4000"]
