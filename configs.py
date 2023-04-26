import time
from datetime import datetime

# dataset = "science"
seed = 24
model_type2path = {
    "electra": "google/electra-base-discriminator",
    "bert": "emilyalsentzer/Bio_ClinicalBERT",# "bert-base-uncased",
    "roberta": "allenai/biomed_roberta_base",
    "albert": "albert-base-v2",
    # "xlm": "",
    # "xlnet": "xlnet-base-cased"
}
model_type = "electra"

mesh_config = dict(
    # paths
    pretrained_path=model_type2path[model_type],
    save_path="./data/models/trained_mesh_" + model_type + datetime.now().strftime("%d_%H:%M:%S") + "/",
    log_path="./data/log/",
    eval_tensor_path="./data/eval/mesh_eval_tensor.json",
    eval_cand_path="./data/eval/mesh_eval_cand.json",
    eval_pos_path="./data/eval/mesh_eval_pos.json",
    # config
    margin_beta=0.2,
    pos_weight=0.4,
    epochs=6,
    batch_size=32,
    try_batch_size=64,
    eval_size=1024,
    lr=2e-5,
    eps=1e-8,
    padding_max=288,
    log_label="t101_w0.4_" + model_type,
    model_type=model_type,
    n_gpu = "3",
    seed = seed,
    eval_sample_num = 21
)
