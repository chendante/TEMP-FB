import temp
import util
import torch
import scorer
import split_data
import configs
from tqdm import tqdm
from functools import partialmethod

tqdm.__init__ = partialmethod(tqdm.__init__, disable=True)


def dag_run(train_args):
    util.set_seed(configs.seed)
    train_args = util.DotDict(train_args)

    trainer = temp.trainer.DAGTrainer(train_args)

    if not trainer.train():
        print("Model didn't converge! Please try another seed.")
        return None
    # trainer.save_model()
    trainer.load_model("best")
    # trainer.eval(100, "test")
    return trainer.eval(819, "test")

def save_eval_data(train_args):
    train_args = util.DotDict(train_args)

    trainer = temp.trainer.DAGTrainer(train_args)
    trainer.save_eval_data()


if __name__ == '__main__':
    print(dag_run(configs.mesh_config))
    # save_eval_data(configs.mesh_config)
    # trainer = temp.trainer.DAGTrainer(util.DotDict(configs.mesh_config))
    # # print(trainer._tokenizer.encode("Hello", add_special_tokens=False))
    # print(trainer._tokenizer.encode_plus("Hello", "Hello", add_special_tokens=True, return_token_type_ids=True))
    # print()
    # print(torch.LongTensor([[torch.LongTensor([i for i in range(j, j+2)]) for j in range(k, k+3)] for k in range(4)]))

# nohup python3.9 run.py > output.file 2>&1 &