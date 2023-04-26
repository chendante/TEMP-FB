import codecs
from util import *
from collections import defaultdict, deque
import networkx as nx
from itertools import product, chain
from networkx.algorithms import descendants


class DAGTaxo:
    def __init__(self) -> None:
        self.term2desc = {}
        self.term2pars = defaultdict(list)
        self.key2term = {}
        self.terms = []
        self.terms_set = set()
        self.full_dag = nx.DiGraph()
        self.train_dag = nx.DiGraph()
        self.mock_root = "mock_root"
        self.end_token = "end"

        # read data
        self.init_terms()
        self.init_desc()
        self.init_taxo()
        self.train_terms, self.test_terms, self.valid_terms = self.init_from_ids(
            DAG_TERM_PATH + ".train"), self.init_from_ids(DAG_TERM_PATH + ".validation"), self.init_from_ids(DAG_TERM_PATH + ".test")
        self.train_terms.add(self.mock_root)

        # build taxonomy dag
        self.build_full_dag()
        self.build_train_dag()

        # build dataset
        self.candidates = self.get_candidates(self.train_dag)
        self.train_node2pos = self.get_node2pos()


    def init_desc(self):
        with codecs.open(DAG_DESC_PATH, 'r') as f:
            for line in f:
                strs = line.strip().split("\t", maxsplit=1)
                if len(strs) == 2:
                    self.term2desc[strs[0]] = strs[1]
                    if strs[0] not in self.terms_set:
                        print(line)
                else:
                    print(line)

    def init_taxo(self):
        with codecs.open(DAG_TAXO_PATH, 'r') as f:
            for line in f:
                strs = line.strip().split("\t", maxsplit=1)
                if len(strs) == 2:
                    self.term2pars[self.key2term[strs[1]]].append(self.key2term[strs[0]])
                else:
                    print(line)

    def init_terms(self):
        with codecs.open(DAG_TERM_PATH, 'r') as f:
            for line in f:
                strs = line.strip().split("\t", maxsplit=1)
                if len(strs) == 2:
                    self.key2term[strs[0]] = strs[1]
                    self.terms.append(strs[1])
                else:
                    print(line)
        self.terms_set = set(self.terms)

    def init_from_ids(self, path):
        res = set()
        with codecs.open(path, 'r') as f:
            for line in f:
                id = int(line.strip()) - 1
                res.add(self.terms[id])
        return res

    def build_full_dag(self):
        for node in self.terms:
            self.full_dag.add_node(node)
        for child, pars in self.term2pars.items():
            for par in pars:
                self.full_dag.add_edge(par, child)
        root_nodes = [node for node in self.full_dag.nodes(
        ) if self.full_dag.in_degree(node) == 0]
        print(len(root_nodes), len(self.full_dag.nodes))
        for n in root_nodes:
            self.full_dag.add_edge(self.mock_root, n)

    def get_subgraph(self, remove_nodes):
        subgraph = self.full_dag.subgraph(
            [node for node in self.full_dag.nodes if node not in remove_nodes]).copy()
        for node in remove_nodes:
            parents = set()
            children = set()
            ps = deque(self.full_dag.predecessors(node))
            cs = deque(self.full_dag.successors(node))
            while ps:
                p = ps.popleft()
                if p in subgraph:
                    parents.add(p)
                else:
                    ps += list(self.full_dag.predecessors(p))
            while cs:
                c = cs.popleft()
                if c in subgraph:
                    children.add(c)
                else:
                    cs += list(self.full_dag.successors(c))
            for p, c in product(parents, children):
                subgraph.add_edge(p, c)
        # remove jump edges
        node2descendants = {n: set(descendants(subgraph, n)) for n in subgraph.nodes}
        for node in subgraph.nodes():
            if subgraph.out_degree(node) > 1:
                successors1 = set(subgraph.successors(node))
                successors2 = set(chain.from_iterable([node2descendants[n] for n in successors1]))
                checkset = successors1.intersection(successors2)
                if checkset:
                    for s in checkset:
                        if subgraph.in_degree(s) > 1:
                            subgraph.remove_edge(node, s)
        return subgraph

    def build_train_dag(self):
        remove_nodes = self.test_terms.union(self.valid_terms)
        subgraph = self.get_subgraph(remove_nodes)
        # # add mock leaf
        # for node in subgraph.nodes():
        #     subgraph.add_edge(node, self.mock_leaf)
        self.train_dag = subgraph

    def get_candidates(self, graph):
        node2descendants = {n: set(descendants(graph, n)) for n in graph.nodes}
        candidates = set(chain.from_iterable([[(n, d) for d in ds] for n, ds in node2descendants.items()]))
        candidates = candidates.union([(n, self.end_token) for n in graph.nodes])
        return list(candidates)

    def find_insert_position(self, nodes, graph):
        node2pos = {}
        for node in nodes:
            parents = set()
            children = set()
            ps = deque(graph.predecessors(node))
            cs = deque(graph.successors(node))
            while ps:
                p = ps.popleft()
                if p not in nodes:
                    parents.add(p)
                else:
                    ps += list(graph.predecessors(p))
            
            while cs:
                c = cs.popleft()
                if c not in nodes:
                    children.add(c)
                else:
                    cs += list(graph.successors(c))
            if not children:
                children.add(self.end_token)
            position = [(p, c) for p in parents for c in children if p != c]
            node2pos[node] = position
        return node2pos
    
    def get_node2pos(self):
        node2pos = {}
        for node in self.train_dag.nodes:
            if node == self.mock_root:
                continue
            node2pos[node] = self.find_insert_position(
                [node], self.train_dag)[node]
        return node2pos

    def get_k_neg_pos(self, node, k):
        res = set()
        while len(res) < k:
            cand = random.choice(self.candidates)
            if cand not in self.train_node2pos[node]:
                res.add(cand)
        return res


if __name__ == "__main__":
    # t = DAGTaxo()
    # print(t.get_k_neg_pos("water deprivation", 10))
    pars = set()
    child = set()
    with codecs.open(DAG_TAXO_PATH, 'r') as f:
            for line in f:
                strs = line.strip().split("\t", maxsplit=1)
                pars.add(strs[0])
                child.add(strs[1])
    print(pars.difference(child))
