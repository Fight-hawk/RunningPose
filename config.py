import yaml


def load_config(config_path='./config.yaml'):
    with open('../config.yaml') as f:
        cfg = yaml.safe_load(f)
    return cfg
