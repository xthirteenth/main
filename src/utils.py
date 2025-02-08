import json
import torch
import random
import numpy as np

def load_config(config_path='config/config.json'):
    with open(config_path, 'r') as f:
        return json.load(f)

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device(config):
    return torch.device(config['device'] if torch.cuda.is_available() else 'cpu')