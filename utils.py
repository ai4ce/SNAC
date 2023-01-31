from munch import Munch
def read_config(config_path):
    with open(config_path, 'r') as f:
        cfg = Munch.fromYAML(f)
    return cfg
