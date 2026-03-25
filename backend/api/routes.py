from fastapi import APIRouter
from .gnn_routes import router as gnn_router
from .generator_routes import router as generator_router
from .vae_routes import router as vae_router
from .spatial_routes import router as spatial_router
from .wl_routes import router as wl_router
from .physics_gnn_routes import router as physics_router
from .hyperbolic_routes import router as hyperbolic_router

api_router = APIRouter(prefix="/api")
api_router.include_router(gnn_router)
api_router.include_router(generator_router)
api_router.include_router(vae_router)
api_router.include_router(spatial_router)
api_router.include_router(wl_router)
api_router.include_router(physics_router)
api_router.include_router(hyperbolic_router)
