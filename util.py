import random
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
import os
import json

now = datetime.now()

REPO_PATH = str(Path(__file__).resolve().parents[0])
DAG_DATA_PATH = REPO_PATH + "/data/mesh/"
DAG_TAXO_PATH = DAG_DATA_PATH + "mesh.taxo"
DAG_DESC_PATH = DAG_DATA_PATH + "mesh.desc"
DAG_TERM_PATH = DAG_DATA_PATH + "mesh.terms"

class DotDict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def set_seed(seed):
    '''
    设置整个开发环境的seed
    :param seed:
    :param device:
    :return:
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

def prepare_device(use_gpu):
    """
    setup GPU device if available, move model into configured device
    # 如果n_gpu_use为数字，则使用range生成list
    # 如果输入的是一个list，则默认使用list[0]作为controller
    Example:
        use_gpu = '' : cpu
        use_gpu = '0': cuda:0
        use_gpu = '0,1' : cuda:0 and cuda:1
     """
    n_gpu_use = [int(x) for x in use_gpu.split(",")] if use_gpu else []
    if not use_gpu:
        device_type = 'cpu'
    else:
        device_type = f"cuda:{n_gpu_use[0]}"
    n_gpu = torch.cuda.device_count()
    if len(n_gpu_use) > 0 and n_gpu == 0:
        print(
            "Warning: There\'s no GPU available on this machine, training will be performed on CPU."
        )
        device_type = 'cpu'
    if len(n_gpu_use) > n_gpu:
        msg = f"Warning: The number of GPU\'s configured to use is {n_gpu}, but only {n_gpu} are available on this machine."
        print(msg)
        n_gpu_use = range(n_gpu)
    device = torch.device(device_type)
    list_ids = n_gpu_use
    return device, list_ids


def model_device(n_gpu, model):
    '''
    :param n_gpu:
    :param model:
    :return:
    '''
    device, device_ids = prepare_device(n_gpu)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    if len(device_ids) == 1:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(device_ids[0])
    model = model.to(device)
    return model, device

def save_json(data, filename):
    """
    存储JSON数据到文件中
    
    Args:
    data: 要存储的JSON数据
    filename: 要存储数据的文件名
    """
    with open(filename, 'w+') as f:
        json.dump(data, f)

def load_json(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    return data
