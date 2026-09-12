import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import product routers from the existing modular structure
from products.email_campaign.api import router as email_campaign_router
from products.churn_predictor.api import router as churn_predictor_router

app = FastAPI(
    title="Core Growth Solutions API",
    description="Multi-product AI Marketing SaaS platform hosting specialized predictive analytics micro-products.",
    version="1.0.0",
    openapi_tags=[
        {
            "name": "Email Campaign Predictor",
            "description": "Multimodal feature extraction & performance prediction for email marketing.",
        },
        {
            "name": "Customer Churn Predictor",
            "description": "Sequence & metric-based customer retention risk prediction.",
        },
        {
            "name": "System",
            "description": "Platform health & operational endpoints.",
        },
    ],
)

# -----------------------------------------------------------------------
# CORS Configuration
# -----------------------------------------------------------------------
# Hardcoded to the real production frontend domain(s). Add more origins to
# this list (comma-separated entries as separate list items) if you deploy
# additional frontends (e.g. a Vercel preview URL) — always as a Python
# list, never a bare string (a bare string gets iterated character by
# character and silently blocks every real origin).
origins = [
    "https://www.coregrowthsolutions.com",
    "https://coregrowthsolutions.com",
    "https://core-growth-solutions.vercel.app",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # always a list — never assign a bare string here
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Router Mounts — strictly aligned with API routing specs
app.include_router(
    email_campaign_router,
    prefix="/api/products/email-campaign-predictor",
    tags=["Email Campaign Predictor"],
)

app.include_router(
    churn_predictor_router,
    prefix="/api/products/churn-predictor",
    tags=["Customer Churn Predictor"],
)


@app.get("/", tags=["System"])
async def root():
    return {
        "platform": "Core Growth Solutions AI Marketing SaaS",
        "status": "online",
        "docs": "/docs",
    }


@app.get("/health", tags=["System"])
async def health_check():
    return {
        "status": "healthy",
        "products": {
            "email_campaign_predictor": "active",
            "churn_predictor": "active",
        },
    }


if __name__ == "__main__":
    # Local development entrypoint only. On Render, the start command should
    # invoke uvicorn directly (e.g. `uvicorn main:app --host 0.0.0.0 --port $PORT`)
    # WITHOUT reload, so this block and its reload=True are never hit in prod.
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
    
