from temp.taxo import TaxStruct
from torch.utils.data import Dataset as TorchDataset
import transformers
import torch
import numpy as np
import random
from data_reader import DAGTaxo
from tqdm import tqdm


class Sampler:
    def __init__(self, tax_graph: TaxStruct):
        self._tax_graph = tax_graph
        self._nodes = list(self._tax_graph.nodes.keys())

    def sampling(self):
        margins = []
        pos_paths = []
        neg_paths = []
        for node, path in self._tax_graph.node2path.items():
            if node == self._tax_graph.root:
                continue
            while True:
                neg_node = random.choice(self._nodes)
                if neg_node != path[1] and neg_node != node:
                    break
            pos_paths.append(path)
            neg_path = [node] + self._tax_graph.node2path[neg_node]
            neg_paths.append(neg_path)
            margins.append(self.margin(path, neg_path))
        return pos_paths, neg_paths, margins

    @staticmethod
    def margin(path_a, path_b):
        com = len(set(path_a).intersection(set(path_b)))
        return max(min((abs(len(path_a) - com) + abs(len(path_b) - com)) / com, 2), 0.5)

class DAGSampler:
    def __init__(self, dag_taxo: DAGTaxo, model_type) -> None:
        self._dag_taxo = dag_taxo
        self.word2des_tokens = {}
        self.word2tokens = {}
        self.need_init_tokens = True
        self._model_type = model_type
        self.sep_times = 2 if model_type == "roberta" else 1

    def sampling(self):
        margins = []
        pos_paths = []
        neg_paths = []
        for term, pos in self._dag_taxo.train_node2pos.items():
            neg_pos = self._dag_taxo.get_k_neg_pos(term, len(pos))
            for p, n in zip(pos, neg_pos):
                pos_paths.append([p[0], term, p[1]])
                neg_paths.append([n[0], term, n[1]])
                margins.append(1)
        return pos_paths, neg_paths, margins

    def init_tokens(self, tokenizer):
        for w, des in tqdm(self._dag_taxo.term2desc.items(), desc="converting des to tokens"):
            des_tokens = tokenizer.encode(des, add_special_tokens=False)
            self.word2des_tokens[w] = des_tokens
        for term in self._dag_taxo.terms:
            self.word2tokens[term] = tokenizer.encode(term, add_special_tokens=False)
        self.word2tokens["end"] = tokenizer.encode("end", add_special_tokens=False)
        self.word2tokens["mock_root"] = tokenizer.encode("medical concept", add_special_tokens=False)
        self.need_init_tokens = False

    def get_eval_samples(self, mode="test"):
        """
        return的 candidates 和 pos_list 用于后续计算score
        """
        if mode == "test":
            queries = self._dag_taxo.test_terms
            remove_nodes = self._dag_taxo.valid_terms
        else:
            queries = self._dag_taxo.valid_terms
            remove_nodes = self._dag_taxo.test_terms
        paths_list = []
        pos_list = []
        test_graph = self._dag_taxo.get_subgraph(remove_nodes=remove_nodes)
        test_node2pos = self._dag_taxo.find_insert_position(
            queries, test_graph)
        if mode == "test":
            candidates = self._dag_taxo.candidates
        else:
            candidates = list(set([pos for q in queries for pos in test_node2pos[q]]))
        for query in tqdm(queries, desc="Gen Eval Samples..."):
            paths = []
            for candidate in candidates:
                paths.append([candidate[0], query, candidate[1]])
            paths_list.append(paths)
            pos_list.append(test_node2pos[query])
        return paths_list, pos_list, candidates


class DAGDataset(TorchDataset):
    def __init__(self, sampler: DAGSampler, tokenizer, word2des, padding_max=256,
                 margin_beta=0.1):
        self._sampler = sampler
        self._word2des = word2des
        self._padding_max = padding_max
        self._margin_beta = margin_beta
        self._tokenizer = tokenizer
        if self._sampler is not None:
            self._pos_paths, self._neg_paths, self._margins = self._sampler.sampling()
        if self._sampler.need_init_tokens:
            self._sampler.init_tokens(tokenizer)
        self._max_len = 256
        self.candidates = None

    def __len__(self):
        return len(self._pos_paths)

    def __getitem__(self, item):
        pos_path = self._pos_paths[item]
        neg_path = self._neg_paths[item]
        margin = self._margins[item]
        pos_ids, pos_type_ids, pos_attn_masks = self.encode_path(pos_path)
        neg_ids, neg_type_ids, neg_attn_masks = self.encode_path(neg_path)
        return dict(pos_ids=pos_ids,
                    neg_ids=neg_ids,
                    pos_type_ids=pos_type_ids,
                    neg_type_ids=neg_type_ids,
                    pos_attn_masks=pos_attn_masks,
                    neg_attn_masks=neg_attn_masks,
                    margin=torch.FloatTensor([margin * self._margin_beta]))

    def encode_path_slow(self, path):
        des_sent = self._sampler.word2des_tokens[path[1]]
        def_sent = str(" " + self._tokenizer.unk_token + " ").join(path)
        encode = self._tokenizer.encode_plus(des_sent, def_sent, add_special_tokens=True, return_token_type_ids=True
                                             # return_tensors='pt'
                                             )
        input_len = len(encode["input_ids"])
        assert input_len <= self._padding_max
        encode["input_ids"] = encode["input_ids"] + [self._tokenizer.pad_token_id] * (self._padding_max - input_len)
        encode["token_type_ids"] = encode["token_type_ids"] + [0] * (self._padding_max - input_len)
        encode["attention_mask"] = encode["attention_mask"] + [0] * (self._padding_max - input_len)
        return torch.LongTensor(encode["input_ids"]), torch.LongTensor(encode["token_type_ids"]), torch.LongTensor(
            encode["attention_mask"])

    def encode_path(self, path):
        des_tokens = self._sampler.word2des_tokens[path[1]]
        path_ids = self._sampler.word2tokens[path[0]].copy()
        for word in path[1:]:
            path_ids += [self._tokenizer.unk_token_id] + self._sampler.word2tokens[word]
        input_ids = [self._tokenizer.cls_token_id] + des_tokens + \
            [self._tokenizer.sep_token_id] * self._sampler.sep_times + path_ids + [self._tokenizer.sep_token_id]
        input_len = len(input_ids)
        self._max_len = max(input_len, self._max_len)
        if self._max_len == input_len:
            print(self._max_len)
        assert input_len <= self._padding_max
        input_ids += [self._tokenizer.pad_token_id] * (self._padding_max - input_len)
        token_type_ids = [0] * int(len(des_tokens)+2) + [1] * int(
            len(path_ids) + self._sampler.sep_times) + [0] * (self._padding_max - input_len)
        if self._sampler.sep_times > 1:
            token_type_ids = [0] * self._padding_max
        attention_mask = [1] * input_len + [0] * (self._padding_max - input_len)
        return torch.LongTensor(input_ids), torch.LongTensor(token_type_ids), torch.LongTensor(attention_mask)


    def yield_eval_data(self, num, mod):
        paths_list, pos_list, candidates = self._sampler.get_eval_samples(mod)
        self.candidates = candidates
        if num < 0 or num >= len(paths_list):
            num = len(paths_list)
        for paths, pos in zip(paths_list[:num], pos_list[:num]):
            yield [self.encode_path(path) for path in paths], pos
        # encoded_samples = [[self.encode_path(path) for path in paths] for paths in tqdm(paths_list[:10], desc="Encoding Path...")]
        # return encoded_samples, candidates, pos_list


class Dataset(TorchDataset):
    def __init__(self, sampler, tokenizer, word2des, padding_max=256,
                 margin_beta=0.1):
        self._sampler = sampler
        self._word2des = word2des
        self._padding_max = padding_max
        self._margin_beta = margin_beta
        self._tokenizer = tokenizer
        if self._sampler is not None:
            self._pos_paths, self._neg_paths, self._margins = self._sampler.sampling()

    def __len__(self):
        return len(self._pos_paths)

    def __getitem__(self, item):
        pos_path = self._pos_paths[item]
        neg_path = self._neg_paths[item]
        margin = self._margins[item]
        pos_ids, pos_type_ids, pos_attn_masks = self.encode_path(pos_path)
        neg_ids, neg_type_ids, neg_attn_masks = self.encode_path(neg_path)
        return dict(pos_ids=pos_ids,
                    neg_ids=neg_ids,
                    pos_type_ids=pos_type_ids,
                    neg_type_ids=neg_type_ids,
                    pos_attn_masks=pos_attn_masks,
                    neg_attn_masks=neg_attn_masks,
                    margin=torch.FloatTensor([margin * self._margin_beta]))

    def encode_path(self, path):
        des_sent = self._word2des[path[0]][0]
        def_sent = str(" " + self._tokenizer.unk_token + " ").join(path)
        encode = self._tokenizer.encode_plus(des_sent, def_sent, add_special_tokens=True, return_token_type_ids=True
                                             # return_tensors='pt'
                                             )
        input_len = len(encode["input_ids"])
        assert input_len <= self._padding_max
        encode["input_ids"] = encode["input_ids"] + [self._tokenizer.pad_token_id] * (self._padding_max - input_len)
        encode["token_type_ids"] = encode["token_type_ids"] + [0] * (self._padding_max - input_len)
        encode["attention_mask"] = encode["attention_mask"] + [0] * (self._padding_max - input_len)
        return torch.LongTensor(encode["input_ids"]), torch.LongTensor(encode["token_type_ids"]), torch.LongTensor(
            encode["attention_mask"])
