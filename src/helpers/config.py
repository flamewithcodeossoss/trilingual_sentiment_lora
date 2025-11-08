from pathlib import Path
import json

DEFAULTS = {
    "model_name": "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    "use_transformers": True,
    # Optional mapping for models that output LABEL_0/1/2
    "label_mapping": {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"},
}


def load_config() -> dict:
    """Load configuration from config.json and merge with defaults.

    Returns a dict with safe defaults if the file is missing or invalid.
    """
    cfg_path = Path(__file__).with_name("config.json")
    data = {}
    if cfg_path.exists():
        try:
            data = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            # Fall back silently to defaults on parse error
            data = {}

    merged = {**DEFAULTS, **data}
    return merged
    