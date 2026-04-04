class ConfigData(BaseModel):
    API_BASE_URL: str
    MODEL_NAME: str
    HF_TOKEN: str

@app.post("/config")
def update_config(cfg: ConfigData):
    import os
    os.environ["API_BASE_URL"] = cfg.API_BASE_URL
    os.environ["MODEL_NAME"] = cfg.MODEL_NAME
    os.environ["HF_TOKEN"] = cfg.HF_TOKEN
    return {"status": "ok"}
