# Binocrypt Cloud Deployment Guide

Binocrypt is pre-configured with multi-platform deployment manifests for **Render**, **Railway**, **Fly.io**, and **Docker**.

---

## Option 1: Render (Recommended Free Cloud Hosting)

Render supports deploying Binocrypt as a single unified service (100% within Render's free tier).

### Step-by-Step Instructions:
1. **Push your code to GitHub**:
   ```bash
   git remote add origin https://github.com/<YOUR_GITHUB_USERNAME>/binocrypt.git
   git push -u origin main
   ```
2. **Log into Render**: Go to [https://dashboard.render.com](https://dashboard.render.com).
3. **Deploy via Blueprint (Automatic)**:
   - Click **New +** -> **Blueprint**.
   - Connect your GitHub repository.
   - Render will detect `render.yaml` and configure:
     - **Service Name**: `binocrypt`
     - **Runtime**: Python 3.11 with Node.js Vite build
     - **Build Command**: `cd frontend && npm install && npm run build && cd ../backend && pip install -r requirements.txt`
     - **Start Command**: `cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT`
     - **Health Check Path**: `/api/health`
   - Click **Apply**.
   - In 2–3 minutes, your live URL will be ready at `https://binocrypt.onrender.com`!

---

## Option 2: Railway

Railway automatically builds and runs the included multi-stage `Dockerfile`.

### Step-by-Step Instructions:
1. Go to [https://railway.app](https://railway.app) and click **New Project**.
2. Select **Deploy from GitHub repo**.
3. Choose your `binocrypt` repository.
4. Railway will detect `Dockerfile` and `railway.toml` and deploy automatically.
5. Under service **Settings** -> **Networking**, click **Generate Domain** to get your public HTTPS URL.

---

## Option 3: Fly.io

Deploy to Fly.io using the included `fly.toml` and `Dockerfile`:

```bash
# Install flyctl if not installed
# curl -L https://fly.io/install.sh | sh

fly auth login
fly launch --no-deploy
fly deploy
```

Fly.io will build the container, attach the health checks, and serve your application with global edge routing.

---

## Option 4: Local or VPS Self-Hosted (Docker Compose)

Run the entire production stack locally or on any cloud VPS (DigitalOcean, Linode, AWS EC2):

```bash
# Build and start container in detached mode
docker compose up -d

# Verify health status
curl http://localhost:8000/api/health

# Open in browser
http://localhost:8000
```

---

## Architecture & Environment Variables

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `PORT` | `8000` | Port for the Uvicorn server |
| `ENVIRONMENT` | `production` | Enables production static file caching |
| `VITE_API_URL` | *(empty)* | Optional: Set only if hosting frontend separately from backend |
| `API_V1_STR` | `/api` | API route prefix |

### Unified Serving Note:
In production, FastAPI serves both:
- **API routes**: `/api/*`
- **Documentation**: `/docs` and `/openapi.json`
- **Frontend SPA**: `/*` (React Single Page Application served from `frontend/dist`)
