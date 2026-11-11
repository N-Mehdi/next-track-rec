from pathlib import Path

from nexttrack.config import load_config, set_dotted

DEFAULT = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"


def test_default_config_loads():
    config = load_config(DEFAULT)
    assert config["data"]["stream"] == "listens"
    assert config["sasrec"]["max_seq_len"] == 200
    assert config["eval"]["k_values"] == [10, 50, 100]


def test_overrides_are_typed():
    config = load_config(DEFAULT, ["sasrec.epochs=3", "sasrec.fp16=false", "data.max_users=500"])
    assert config["sasrec"]["epochs"] == 3
    assert config["sasrec"]["fp16"] is False
    assert config["data"]["max_users"] == 500


def test_set_dotted_creates_sections():
    config = {}
    set_dotted(config, "a.b.c", 1)
    assert config == {"a": {"b": {"c": 1}}}
