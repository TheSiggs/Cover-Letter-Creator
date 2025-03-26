# Use a full Debian-based Python image for compatibility with Playwright
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install required system dependencies for Playwright
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget gnupg ca-certificates curl unzip fonts-liberation libnss3 \
    libatk-bridge2.0-0 libgtk-3-0 libxss1 libasound2 libxcomposite1 \
    libxrandr2 libxdamage1 libgbm1 libpango-1.0-0 libpangocairo-1.0-0 \
    libx11-xcb1 libxext6 libxfixes3 libxi6 libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --no-cache-dir poetry

# Copy Poetry config and install dependencies
COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.create false \
    && poetry install --no-root --no-interaction --no-ansi

# Install Playwright browsers
RUN pip install --no-cache-dir playwright \
    && playwright install --with-deps

# Copy the rest of the app
COPY . .

# Expose any relevant port (adjust if needed)
EXPOSE 8000

# Run the app
CMD ["python", "api.py"]

