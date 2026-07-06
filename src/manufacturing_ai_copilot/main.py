from fastapi import FastAPI
from manufacturing_ai_copilot.core.config import SERVICE_NAME, VERSION

app = FastAPI(title=SERVICE_NAME, version=VERSION)


@app.get("/health")
def health_check() -> dict[str, str]:
    return {
        "status": "ok",
        "service": SERVICE_NAME,
        "version": VERSION,
    }
