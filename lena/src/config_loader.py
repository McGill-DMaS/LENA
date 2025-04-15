import toml
import os

CONFIG_PATH = './config.toml'

def load_config(config_path=CONFIG_PATH):
    with open(config_path, 'r') as config_file:
        config = toml.load(config_file)
    return config

config = load_config()
#print(config)
