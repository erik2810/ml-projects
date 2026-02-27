from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path

from backend.api.routes import api_router

app = FastAPI(
    title="Graph ML Lab",
    description="Interactive demos for GNNs, graph generation, and graph VAEs — all from scratch.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(api_router)

# serve frontend
frontend_dir = Path(__file__).resolve().parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/static", StaticFiles(directory=frontend_dir / "static"), name="static")

    @app.get("/")
    async def index():
        return FileResponse(frontend_dir / "index.html")
