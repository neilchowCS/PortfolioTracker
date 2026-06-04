"""Persistent config for retirement planner defaults."""
import json
from pathlib import Path

_CONFIG_PATH = Path(__file__).resolve().parent.parent / "retirement_defaults.json"

_DEFAULTS = {
    "expected_return": 5.0,
    "current_age": 30,
    "retirement_age": 50,
    "lifespan": 90,
    "annual_income": 100000.0,
    "withdrawal_mode": "Rate (%)",
    "withdrawal_rate": 3.5,
    "withdrawal_amount": 60000.0,
    "ss_annual": 0.0,
    "ss_start_age": 67,
    "cost_basis_pct": 50,
    "contrib_taxable": 0.0,
    "contrib_pretax": 30500.0,
    "contrib_roth": 53400.0,
    "contrib_aftertax": 0.0,
}


def load_retirement_defaults() -> dict:
    """Load saved defaults, falling back to built-in defaults."""
    cfg = dict(_DEFAULTS)
    if _CONFIG_PATH.exists():
        try:
            with open(_CONFIG_PATH) as f:
                saved = json.load(f)
            cfg.update(saved)
        except (json.JSONDecodeError, OSError):
            pass
    return cfg


def save_retirement_defaults(cfg: dict) -> None:
    """Save current values as defaults."""
    with open(_CONFIG_PATH, "w") as f:
        json.dump(cfg, f, indent=2)
