from fastapi import APIRouter
from .gnn_routes import router as gnn_router
from .generator_routes import router as generator_router
from .vae_routes import router as vae_router
from .spatial_routes import router as spatial_router

api_router = APIRouter(prefix="/api")
api_router.include_router(gnn_router)
api_router.include_router(generator_router)
api_router.include_router(vae_router)
api_router.include_router(spatial_router)
