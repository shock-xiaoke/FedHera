import random
import os
import numpy as np
import torch

def seed_torch(seed, deterministic=False):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = not deterministic
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = not deterministic
    torch.backends.cudnn.allow_tf32 = not deterministic

from .model_aggregation import FedAvg, FlexLoRA, HetLoRA, truncate, FedHera, FedHeLLo, FLoRA, FedHL ,TRAFFIC_STATS
from .client_participation_scheduling import client_selection
from .client import GeneralClient
from .adaptive_peft import (seed_torch, tokenize, load_weight_local, load_weight_hetlora, distribute_weight_fast, modify_adapter,
                    distribute_weight)
