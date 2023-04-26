from typing import Dict
import torch
import temp.sampler as sampler
from torch.utils.data import dataloader
import transformers
from transformers import BertTokenizer, ElectraTokenizer, AlbertTokenizer, RobertaTokenizer, XLMTokenizer, \
    XLNetTokenizer
from tqdm import tqdm
from temp.base import BaseTrainer
import codecs
from temp import TEMPBert, TEMPElectra, TEMPAlbert, TEMPRoberta, TEMPAutoRoberta, TEMPXLNet, TEMPXLM, weighted_loss_fct
import json
import data_reader
import util
import more_itertools as mit
import scorer
import os


class DAGTrainer(BaseTrainer):
    models = {'bert': TEMPBert, 'electra': TEMPElectra, 'albert': TEMPAlbert, 'roberta': TEMPAutoRoberta,
              'xlnet': TEMPXLNet, 'xlm': TEMPXLM}
    tokenizers = {'bert': BertTokenizer, 'electra': ElectraTokenizer, 'albert': AlbertTokenizer,
                  'roberta': RobertaTokenizer, 'xlnet': XLNetTokenizer, 'xlm': XLMTokenizer}

    def __init__(self, args):
        super().__init__(args)
        self.dag_taxo = data_reader.DAGTaxo()
        self.sampler = sampler.DAGSampler(self.dag_taxo, args.model_type)
        self.model = self.models[args.model_type].from_pretrained(args.pretrained_path,
                                                                  # gradient_checkpointing=True,
                                                                  output_attentions=False,
                                                                  output_hidden_states=False,
                                                                  cache_dir="./cache/",
                                                                  # force_download=True
                                                                  )
        self.model, self.device = util.model_device(self.args.n_gpu, self.model)
        self._tokenizer = self.tokenizers[args.model_type].from_pretrained(self.args.pretrained_path,
                                                                           cache_dir="./cache/")
        self.eval_it = 0
        self.cached_models = {}
    
    def train(self):
        optimizer = transformers.AdamW(self.model.parameters(),
                                       lr=self.args.lr,  # args.learning_rate - default is 5e-5
                                       eps=self.args.eps  # args.adam_epsilon  - default is 1e-8
                                       )
        loss_count = 0
        lite_loss_count = 0
        loss_max = 0
        train_samples = {}
        eval_max = 0
        for epoch in tqdm(range(self.args.epochs), desc="Training", total=self.args.epochs):
            dataset = sampler.DAGDataset(self.sampler,
                                  tokenizer=self._tokenizer,
                                  word2des=self.dag_taxo.term2desc,
                                  padding_max=self.args.padding_max,
                                  margin_beta=self.args.margin_beta)
            data_loader = dataloader.DataLoader(dataset, batch_size=self.args.try_batch_size, shuffle=True, drop_last=True)
            loss_all = 0.0
            for batch in data_loader:
                optimizer.zero_grad()
                train_batch, lite_loss = self.get_train_samples(batch)
                loss_all += lite_loss
                self.log_tensorboard(lite_loss, lite_loss_count, "", "lite_loss")
                lite_loss_count += 1
                train_samples = self.cat_samples([train_samples, train_batch], self.args.batch_size)
                if len(train_samples["pos_ids"]) >= self.args.batch_size:
                    # able to train
                    self.model.train()
                    batch = train_samples
                    pos_output = self.model(input_ids=batch["pos_ids"].to(self.device), token_type_ids=batch["pos_type_ids"].to(self.device),
                                            attention_mask=batch["pos_attn_masks"].to(self.device))
                    neg_output = self.model(input_ids=batch["neg_ids"].to(self.device), token_type_ids=batch["neg_type_ids"].to(self.device),
                                            attention_mask=batch["neg_attn_masks"].to(self.device))
                    loss = weighted_loss_fct(pos_output, neg_output, batch["margin"].to(self.device), self.args.pos_weight)
                    loss.backward()
                    train_samples = {}
                    self.log_tensorboard(loss.item(), loss_count, "")
                    loss_count += 1
                    optimizer.step()
            if epoch >= 1:
                results = self.eval(self.args.eval_sample_num, "test")
                if results["hit_at_1"] + results["precision_at_1"] > eval_max:
                    eval_max = results["hit_at_1"] + results["precision_at_1"]
                    self.save_model("best")
            loss_max = max(loss_max, loss_all)
            # if epoch > 4 and loss_all > loss_max * 0.3:
            #     return False  # not converge
        return True

    def get_train_samples(self, batch: Dict):
        self.model.eval()
        loss = 0
        with torch.no_grad():
            pos_output = self.model(input_ids=batch["pos_ids"].to(self.device), token_type_ids=batch["pos_type_ids"].to(self.device),
                                    attention_mask=batch["pos_attn_masks"].to(self.device))
            neg_output = self.model(input_ids=batch["neg_ids"].to(self.device), token_type_ids=batch["neg_type_ids"].to(self.device),
                                    attention_mask=batch["neg_attn_masks"].to(self.device))
            pos_output = pos_output.view(-1).cpu()
            neg_output = neg_output.view(-1).cpu()
            wanted_idx = [i for i in range(pos_output.shape[0]) if pos_output[i]-neg_output[i]<batch["margin"][i]]
            loss = sum([max(batch["margin"][i]-pos_output[i]+neg_output[i], 0) for i in range(pos_output.shape[0])])
        res = {}
        for key in batch.keys():
            res[key] = batch[key][wanted_idx]
        return res, loss

    @staticmethod
    def cat_samples(right_samples_list, max_size):
        """
        将一个list的right_samples按照dict内tensor做合并，并设定合并后的样本最多为max_size个。
        
        参数：
        - right_samples_list：包含多个字典的列表，每个字典表示一个符合条件的样本集合。
        - max_size：合并后最多的样本数。
        
        返回值：
        - 一个新的字典，表示合并后的样本集合。
        """
        # 创建一个空字典，用于保存合并后的样本
        cat_samples = {}
        
        # 遍历right_samples_list，将所有字典内的tensor合并
        for right_samples in right_samples_list:
            for key, value in right_samples.items():
                if key in cat_samples:
                    cat_samples[key] = torch.cat([cat_samples[key], value], dim=0)
                else:
                    cat_samples[key] = value.clone()
        
        # 如果合并后的样本数超过了max_size，则随机选择一部分样本
        if len(cat_samples[list(cat_samples.keys())[0]]) > max_size:
            indices = torch.randperm(len(cat_samples[list(cat_samples.keys())[0]])).tolist()
            indices = indices[:max_size]
            for key, value in cat_samples.items():
                cat_samples[key] = value[indices]
        
        return cat_samples

    def eval(self, num_sample, mod="test"):
        print("BEGIN EVAL")
        self.model.eval()
        all_ranks = []
        dataset = sampler.DAGDataset(self.sampler,
                                  tokenizer=self._tokenizer,
                                  word2des=self.dag_taxo.term2desc,
                                  padding_max=self.args.padding_max,
                                  margin_beta=self.args.margin_beta)
        with torch.no_grad():
            for node_samples, node_pos in tqdm(dataset.yield_eval_data(num_sample, mod), desc="Evaluating..."):
                node_scores = []
                for input_ids, token_type_ids, attention_mask in self.batched_tensor(node_samples, self.args.eval_size):
                    scores = self.model(
                        input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
                    node_scores.append(scores)
                node_scores = torch.cat(node_scores)
                node_scores, labels = scorer.rearrange(node_scores, dataset.candidates, node_pos)
                all_ranks.extend(scorer.obtain_ranks(node_scores, labels))
            total_metrics = scorer.all_metrics(all_ranks)
        self.log_tensor_json(total_metrics, self.eval_it, "")
        self.eval_it += 1
        total_metrics["mod"] = mod
        total_metrics["num_sample"] = num_sample
        self.log_json(total_metrics)
        return total_metrics

    def load_eval_data(self):
        encoded_samples, candidates, pos_list = torch.load(self.args.eval_tensor_path), util.load_json(
            self.args.eval_cand_path), util.load_json(self.args.eval_pos_path)
        return encoded_samples, candidates, pos_list

    def batched_tensor(self, node_samples, batch_size):
        for samples in mit.chunked(node_samples, batch_size):
            input_ids, token_type_ids, attention_mask = [], [], []
            for sample in samples:
                input_ids.append(sample[0])
                token_type_ids.append(sample[1])
                attention_mask.append(sample[2])
            yield torch.stack(input_ids).to(self.device), torch.stack(token_type_ids).to(self.device), torch.stack(attention_mask).to(self.device)


    def save_model(self, label=""):
        if self.args.model_type == "roberta":
            self.cached_models[label] = self.models[self.args.model_type].from_pretrained(self.args.pretrained_path,
                                                                                     # gradient_checkpointing=True,
                                                                                     output_attentions=False,
                                                                                     output_hidden_states=False,
                                                                                     cache_dir="./cache/",
                                                                                     # force_download=True
                                                                                     )
            self.cached_models[label].load_state_dict(self.model.state_dict())
            return
        self.model.save_pretrained(self.args.save_path + label)
        self._tokenizer.save_pretrained(self.args.save_path + label)

    def load_model(self, label="best"):
        if not os.path.isdir(self.args.save_path + label):
            return
        if self.args.model_type == "roberta":
            self.model = self.cached_models[label]
            # self.model = torch.load(self.args.save_path + label)
            self.model = self.model.to(self.device)
            return
        self.model = self.models[self.args.model_type].from_pretrained(
            self.args.save_path + label)
        self.model = self.model.to(self.device)



class Trainer(BaseTrainer):
    models = {'bert': TEMPBert, 'electra': TEMPElectra, 'albert': TEMPAlbert, 'roberta': TEMPAutoRoberta,
              'xlnet': TEMPXLNet, 'xlm': TEMPXLM}
    tokenizers = {'bert': BertTokenizer, 'electra': ElectraTokenizer, 'albert': AlbertTokenizer,
                  'roberta': RobertaTokenizer, 'xlnet': XLNetTokenizer, 'xlm': XLMTokenizer}

    def __init__(self, args):
        super().__init__(args)
        with codecs.open(args.taxo_path, encoding='utf-8') as f:
            # TAXONOMY FILE FORMAT: hypernym <TAB> term
            tax_lines = f.readlines()
        tax_pairs = [line.strip().split("\t") for line in tax_lines]
        self.tax_graph = sampler.TaxStruct(tax_pairs)
        self.sampler = sampler.Sampler(self.tax_graph)

        self.model = self.models[args.model_type].from_pretrained(args.pretrained_path,
                                                                  # gradient_checkpointing=True,
                                                                  output_attentions=False,
                                                                  output_hidden_states=False,
                                                                  cache_dir="./cache/",
                                                                  # force_download=True
                                                                  )
        self._tokenizer = self.tokenizers[args.model_type].from_pretrained(self.args.pretrained_path,
                                                                           cache_dir="./cache/")
        with open(args.dic_path, 'r', encoding='utf-8') as fp:
            self._word2des = json.load(fp)

    def train(self):
        optimizer = transformers.AdamW(self.model.parameters(),
                                       lr=self.args.lr,  # args.learning_rate - default is 5e-5
                                       eps=self.args.eps  # args.adam_epsilon  - default is 1e-8
                                       )
        dataset = sampler.Dataset(self.sampler,
                                  tokenizer=self._tokenizer,
                                  word2des=self._word2des,
                                  padding_max=self.args.padding_max,
                                  margin_beta=self.args.margin_beta)
        data_loader = dataloader.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
        scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                                 num_warmup_steps=0,
                                                                 num_training_steps=len(data_loader) * self.args.epochs)
        self.model.cuda()
        loss_count = 0
        loss_max = 0
        for epoch in tqdm(range(self.args.epochs), desc="Training", total=self.args.epochs):
            dataset = sampler.Dataset(self.sampler,
                                      tokenizer=self._tokenizer,
                                      word2des=self._word2des,
                                      padding_max=self.args.padding_max,
                                      margin_beta=self.args.margin_beta)
            data_loader = dataloader.DataLoader(dataset, batch_size=self.args.batch_size, shuffle=True, drop_last=True)
            loss_all = 0.0
            for batch in data_loader:
                optimizer.zero_grad()
                pos_output = self.model(input_ids=batch["pos_ids"].cuda(), token_type_ids=batch["pos_type_ids"].cuda(),
                                        attention_mask=batch["pos_attn_masks"].cuda())
                neg_output = self.model(input_ids=batch["neg_ids"].cuda(), token_type_ids=batch["neg_type_ids"].cuda(),
                                        attention_mask=batch["neg_attn_masks"].cuda())
                loss = self.model.margin_loss_fct(pos_output, neg_output, batch["margin"].cuda())
                loss.backward()
                optimizer.step()
                scheduler.step()
                loss_all += loss.item()
                self._log_tensorboard(self.args.log_label, "", loss.item(), loss_count)
                loss_count += 1
            loss_max = max(loss_max, loss_all)
            if epoch > 4 and loss_all > loss_max * 0.3:
                return False  # not converge
        return True

    def save_model(self):
        self.model.save_pretrained(self.args.save_path)
        self._tokenizer.save_pretrained(self.args.save_path)
