import json
from pathlib import Path
from typing import Any, Dict


def load_config_file(path: str) -> Dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    suffix = config_path.suffix.lower()
    text = config_path.read_text(encoding="utf-8")

    if suffix == ".json":
        data = json.loads(text)
    elif suffix in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "YAML config requires PyYAML. Install with: pip install pyyaml"
            ) from exc
        data = yaml.safe_load(text)
    else:
        raise ValueError(f"Unsupported config format: {suffix}. Use .json/.yml/.yaml")

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError("Top-level config must be a key-value object")
    return data

