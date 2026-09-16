# ==========================================
# STAGE 1: Build Frontend Single Page App
# ==========================================
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend

COPY frontend/package*.json ./
RUN npm ci --prefer-offline --no-audit

COPY frontend/ ./
RUN npm run build

# ==========================================
# STAGE 2: Python Backend & Static Server
# ==========================================
FROM python:3.11-slim AS production

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY backend/requirements.txt ./backend/
RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -r ./backend/requirements.txt

# Copy backend application source
COPY backend/ ./backend/

# Copy built frontend assets to the expected static directory
COPY --from=frontend-builder /app/frontend/dist ./frontend/dist

WORKDIR /app/backend
RUN chmod +x ./start.sh && \
    ln -s /app/backend /app/backend/backend && \
    ln -s /app/backend/start.sh /usr/local/bin/start-binocrypt

EXPOSE 8000

CMD ["/app/backend/start.sh"]


