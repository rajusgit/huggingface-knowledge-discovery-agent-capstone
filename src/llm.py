

def _get_config() -> AppConfig:
    cfg = load_config(os.path.join(_PROJECT_ROOT, "config", "config.yaml"))
    # Screener batch scores via OpenRouter google/gemini-3.5-flash-lite (llm.localllm block).
    cfg.llm.provider = "localllm"
    return cfg