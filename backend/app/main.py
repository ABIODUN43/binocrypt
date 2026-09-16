import os
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from .core.config import settings
from .core.database import init_db
from .api import market, scanner, setups, risk, paper, backtest, explain, intelligence, research
from .services.paper_trader import paper_trader

background_trigger_task = None

async def periodic_trigger_watcher():
    """Checks open paper positions every 10 seconds for TP/SL hits."""
    while True:
        try:
            await asyncio.sleep(10)
            await paper_trader.check_triggers()
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[TriggerWatcher] Error: {e}")
            await asyncio.sleep(10)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("[Binocrypt] Initializing database...")
    await init_db()
    print("[Binocrypt] Database initialized.")
    
    # Start background watcher
    global background_trigger_task
    background_trigger_task = asyncio.create_task(periodic_trigger_watcher())
    print("[Binocrypt] Paper trading trigger watcher started.")
    
    yield
    
    # Shutdown
    if background_trigger_task:
        background_trigger_task.cancel()
        try:
            await background_trigger_task
        except asyncio.CancelledError:
            pass
    print("[Binocrypt] Shutdown complete.")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Institutional-grade quantitative crypto market-scanning and decision-support system.",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(market.router, prefix=settings.API_V1_STR)
app.include_router(scanner.router, prefix=settings.API_V1_STR)
app.include_router(setups.router, prefix=settings.API_V1_STR)
app.include_router(risk.router, prefix=settings.API_V1_STR)
app.include_router(paper.router, prefix=settings.API_V1_STR)
app.include_router(backtest.router, prefix=settings.API_V1_STR)
app.include_router(explain.router, prefix=settings.API_V1_STR)
app.include_router(intelligence.router, prefix=settings.API_V1_STR)
app.include_router(research.router, prefix=settings.API_V1_STR)

@app.get("/api/health")
@app.get("/healthz")
@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "system": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "documentation": "/docs"
    }

# Unified Single-Service Deployment: Serve production Vite frontend if dist exists
frontend_dist = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "frontend", "dist"))
if os.path.exists(frontend_dist) and os.path.isdir(frontend_dist):
    assets_dir = os.path.join(frontend_dist, "assets")
    if os.path.exists(assets_dir):
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        # Don't intercept API or docs routes
        if full_path.startswith("api/") or full_path == "api" or full_path.startswith("docs") or full_path.startswith("openapi.json"):
            raise HTTPException(status_code=404, detail="API route not found")
        file_path = os.path.join(frontend_dist, full_path)
        if full_path and os.path.isfile(file_path):
            return FileResponse(file_path)
        index_file = os.path.join(frontend_dist, "index.html")
        if os.path.exists(index_file):
            return FileResponse(index_file)
        raise HTTPException(status_code=404, detail="File not found")
else:
    @app.get("/")
    async def root():
        return {
            "status": "online",
            "system": settings.PROJECT_NAME,
            "version": settings.VERSION,
            "documentation": "/docs"
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)

