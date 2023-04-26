import itertools
import re

import numpy as np
import torch

def score(results, tax_graph, trues):
    def get_wu_p(node_a, node_b):
        # if node_a == node_b:
        #     return 1.0
        full_path_a = node2full_path[node_a]
        full_path_b = node2full_path[node_b]
        com = full_path_a.intersection(full_path_b)
        lca_dep = 1
        for node in com:
            lca_dep = max(len(tax_graph.node2path[node]), lca_dep)
        dep_a = len(tax_graph.node2path[node_a])
        dep_b = len(tax_graph.node2path[node_b])
        res = 2.0 * float(lca_dep) / float(dep_a + dep_b)
        # assert res <= 1
        return res

    node2full_path = tax_graph.get_node2full_path()
    wu_p, acc, mrr = 0.0, 0.0, 0.0
    wrong_set = []
    ii = 0
    for result, ground_true in zip(results, trues):
        if result[0] == ground_true:
            acc += 1
        else:
            wrong_set.append([ii, result[0], ground_true])
        num = 0
        for i, r in enumerate(result):
            if r == ground_true:
                num = i + 1.0
                break
        mrr += 1.0 / num
        wu_p += get_wu_p(result[0], ground_true)
        ii += 1
    acc /= float(len(results))
    mrr /= float(len(results))
    wu_p /= float(len(results))

    return acc, mrr, wu_p, wrong_set

def rearrange(energy_scores, candidate_position, true_position):
    tmp = np.array([[x == y for x in candidate_position] for y in true_position]).any(0)
    correct = np.where(tmp)[0]
    incorrect = np.where(~tmp)[0]
    labels = torch.cat((torch.ones(len(correct)), torch.zeros(len(incorrect)))).int()
    energy_scores = torch.cat((energy_scores[correct], energy_scores[incorrect]))
    return energy_scores, labels

def calculate_ranks_from_similarities(all_similarities, positive_relations):
    """
    all_similarities: a np array
    positive_relations: a list of array indices

    return a list
    """
    # positive_relation_similarities = all_similarities[positive_relations]
    # negative_relation_similarities = np.ma.array(all_similarities, mask=False)
    # negative_relation_similarities.mask[positive_relations] = True
    # ranks = list((negative_relation_similarities > positive_relation_similarities[:, np.newaxis]).sum(axis=1) + 1)
    # ranks = list((all_similarities > positive_relation_similarities[:, np.newaxis]).sum(axis=1) + 1)
    ranks = list(np.argsort(np.argsort(-all_similarities))[positive_relations] + 1)
    return ranks


def obtain_ranks(outputs, targets):
    """ 
    outputs : tensor of size (batch_size, 1), required_grad = False, model predictions
    targets : tensor of size (batch_size, ), required_grad = False, labels
        Assume to be of format [1, 0, ..., 0, 1, 0, ..., 0, ..., 0]
    mode == 0: rank from distance (smaller is preferred)
    mode == 1: rank from similarity (larger is preferred)
    """

    calculate_ranks = calculate_ranks_from_similarities
    all_ranks = []
    prediction = outputs.cpu().numpy().squeeze()
    label = targets.cpu().numpy()
    sep = np.array([0, 1], dtype=label.dtype)

    # fast way to find subarray indices in a large array, c.f. https://stackoverflow.com/questions/14890216/return-the-indexes-of-a-sub-array-in-an-array
    end_indices = [(m.start() // label.itemsize) + 1 for m in re.finditer(sep.tostring(), label.tostring())]
    end_indices.append(len(label) + 1)
    start_indices = [0] + end_indices[:-1]
    for start_idx, end_idx in zip(start_indices, end_indices):
        distances = prediction[start_idx: end_idx]
        labels = label[start_idx:end_idx]
        positive_relations = list(np.where(labels == 1)[0])
        ranks = calculate_ranks(distances, positive_relations)
        all_ranks.append(ranks)
    return all_ranks

def macro_mr(all_ranks):
    macro_mr = np.array([np.array(all_rank).mean() for all_rank in all_ranks]).mean()
    return macro_mr


def micro_mr(all_ranks):
    micro_mr = np.array(list(itertools.chain(*all_ranks))).mean()
    return micro_mr


def hit_at_1(all_ranks):
    rank_positions = np.array(list(itertools.chain(*all_ranks)))
    hits = np.sum(rank_positions <= 1)
    return 1.0 * hits / len(rank_positions)


def hit_at_3(all_ranks):
    rank_positions = np.array(list(itertools.chain(*all_ranks)))
    hits = np.sum(rank_positions <= 3)
    return 1.0 * hits / len(rank_positions)


def hit_at_5(all_ranks):
    rank_positions = np.array(list(itertools.chain(*all_ranks)))
    hits = np.sum(rank_positions <= 5)
    return 1.0 * hits / len(rank_positions)


def hit_at_10(all_ranks):
    rank_positions = np.array(list(itertools.chain(*all_ranks)))
    hits = np.sum(rank_positions <= 10)
    return 1.0 * hits / len(rank_positions)


def precision_at_1(all_ranks):
    rank_positions = np.array(list(itertools.chain(*all_ranks)))
    hits = np.sum(rank_positions <= 1)
    return 1.0 * hits / len(all_ranks)


def precision_at_3(all_ranks):
    rank_positions = np.array(list(itertools.chain(*all_ranks)))
    hits = np.sum(rank_positions <= 3)
    return 1.0 * hits / (len(all_ranks) * 3)


def precision_at_5(all_ranks):
    rank_positions = np.array(list(itertools.chain(*all_ranks)))
    hits = np.sum(rank_positions <= 5)
    return 1.0 * hits / (len(all_ranks) * 5)


def precision_at_10(all_ranks):
    rank_positions = np.array(list(itertools.chain(*all_ranks)))
    hits = np.sum(rank_positions <= 10)
    return 1.0 * hits / (len(all_ranks) * 10)


def mrr_scaled_10(all_ranks):
    """ Scaled MRR score, check eq. (2) in the PinSAGE paper: https://arxiv.org/pdf/1806.01973.pdf
    """
    rank_positions = np.array(list(itertools.chain(*all_ranks)))
    scaled_rank_positions = np.ceil(rank_positions / 10)
    return (1.0 / scaled_rank_positions).mean()


def all_metrics(all_ranks):
    return {"macro_mr": macro_mr(all_ranks), "micro_mr": micro_mr(all_ranks), "hit_at_1": hit_at_1(all_ranks), "hit_at_3": hit_at_3(all_ranks), "hit_at_5": hit_at_5(all_ranks), "hit_at_10": hit_at_10(all_ranks), "precision_at_1": precision_at_1(all_ranks), "precision_at_3": precision_at_3(all_ranks), "precision_at_5": precision_at_5(all_ranks), "precision_at_10": precision_at_10(all_ranks), "mrr_scaled_10": mrr_scaled_10(all_ranks)}
