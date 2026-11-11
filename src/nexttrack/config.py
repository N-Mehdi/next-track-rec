"""YAML configuration with dotted command line overrides."""

import argparse
import copy

import yaml


def load_config(path, overrides=None):
    """Load a YAML file and apply overrides of the form section.key=value."""
    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if overrides is None:
        overrides = []
    for item in overrides:
        key, raw_value = item.split("=", 1)
        set_dotted(config, key, parse_value(raw_value))
    return config


def set_dotted(config, dotted_key, value):
    """Set config["a"]["b"] from the key "a.b", creating sections as needed."""
    parts = dotted_key.split(".")
    node = config
    for part in parts[:-1]:
        if part not in node or not isinstance(node[part], dict):
            node[part] = {}
        node = node[part]
    node[parts[-1]] = value


def parse_value(raw):
    """Parse an override value with YAML rules (ints, floats, bools, lists)."""
    return yaml.safe_load(raw)


def copy_config(config):
    return copy.deepcopy(config)


def parse_args(description, add_arguments=None):
    """Common command line: --config path and --set section.key=value overrides."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    if add_arguments is not None:
        add_arguments(parser)
    args = parser.parse_args()
    return load_config(args.config, args.set), args
