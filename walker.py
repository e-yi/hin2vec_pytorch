import random
from itertools import product
import networkx as nx
from collections import defaultdict


class HIN:
    """
    Class to generate vertex sequences.
    """

    def __init__(self, window=None):
        print("Model initialization started.")
        self.graph = nx.DiGraph()
        self.node_size = 0
        self.path_size = 0

        def new_id():
            i = self.node_size
            self.node_size += 1
            return i

        self.node2id = defaultdict(new_id)
        self.id2type = {}
        self._window = window
        self.node_types = set()
        self.path2id = None
        self.id2node = None

    @property
    def window(self):
        return self._window

    @window.setter
    def window(self, val):
        if not self._window:
            self._window = val
        else:
            raise ValueError("window只能被设定一次")

    def add_edge(self, source_node, source_class, dest_node, dest_class, edge_class, weight):
        i = self.node2id[source_node]
        j = self.node2id[dest_node]
        self.id2type[i] = source_class
        self.id2type[j] = dest_class
        self.node_types.add(source_class)
        self.node_types.add(dest_class)
        self.graph.add_edge(i, j, weight=weight)

    def small_walk(self, start_node, length):
        walk = [start_node]
        for i in range(1, length):
            if next(nx.neighbors(self.graph, walk[-1]), None) is None:
                break
            walk += random.sample(list(nx.neighbors(self.graph, walk[-1])), 1)  # todo 添加按权重游走的采样方式
        return walk

    def do_walks(self, length):
        for start_node in range(self.node_size):
            yield self.small_walk(start_node, length)

    def sample(self, length):
        """
        从随机游走的结果中截取结果返回，每个节点轮流作为起始节点一次
        :param length: 游走长度
        :return: （start_id,end_id,edge_type)
        """
        if not self.window:
            raise ValueError("window not set")

        if not self.path2id:
            self.path2id = {}
            path_id = 0
            for w in range(1, self._window + 1):
                for i in product(self.node_types, repeat=w + 1):
                    self.path2id[i] = path_id
                    path_id += 1

            self.path_size = len(self.path2id)
            self.id2node = {v: k for k, v in self.node2id.items()}

        samples = []

        for walk in self.do_walks(length):
            cur_len = 0
            for i, node_id in enumerate(walk):
                cur_len = min(cur_len + 1, self._window + 1)  # 当window=n的时候，最长路径有n+1个节点
                if cur_len >= 2:
                    for path_length in range(1, cur_len):
                        sample = (walk[i - path_length], walk[i],
                                  self.path2id[tuple([self.id2type[t] for t in walk[i - path_length:i + 1]])])
                        # print(tuple([self.id2type[t] for t in walk[i-path_length:i + 1]]))
                        samples.append(sample)

        return samples

    def print_statistics(self):
        print(f'size = {self.node_size}')

def format_data(sample, path_size, neg=5):
    """
    完全随机的负采样 todo 改进一下
    :param path_size: 元路径总数
    :param neg: 负采样数目
    :param sample: HIN.sample()返回值，(start_node, end_node, path_id)
    :return: data_entry ((start_node, end_node, path_id), true or false) true -> 1 false -> 0
    """
    for start_node, end_node, path_id in sample:
        yield (start_node, end_node, path_id), 1

        for i in range(neg):
            yield (start_node, end_node, random.randint(0, path_size-1)), 0

if __name__ == '__main__':
    hin = HIN()
    hin.window = 4
    # hin.window = 6
    # hin.add_edge('A', 'Dr', 'a', 'Di', None, 0.3)
    # hin.add_edge('B', 'Dr', 'b', 'Di', None, 0.3)
    # hin.add_edge('C', 'Dr', 'c', 'Di', None, 0.3)
    # hin.add_edge('A', 'Dr', 'b', 'Di', None, 0.3)
    # hin.add_edge('C', 'Dr', 'b', 'Di', None, 0.3)
    # hin.add_edge('c', 'Di', 'A', 'Dr', None, 0.3)
    # hin.add_edge('a', 'Di', 'B', 'Dr', None, 0.3)
    # hin.add_edge('A', 'Dr', 'B', 'Dr', None, 0.3)

    hin.add_edge('A', 'Dr', 'B', 'Di', None, 0.3)
    hin.add_edge('B', 'Di', 'C', 'Dr', None, 0.3)
    hin.add_edge('C', 'Dr', 'D', 'Di', None, 0.3)
    hin.add_edge('D', 'Di', 'E', 'Dr', None, 0.3)
    hin.add_edge('E', 'Dr', 'F', 'Di', None, 0.3)
    hin.add_edge('F', 'Di', 'A', 'Dr', None, 0.3)

    print(hin.small_walk(hin.node2id['A'], 4))
    print(hin.sample(3))
    print(hin.node_size)
    print(hin.path_size)

    print(hin.graph.edges)

    print(list(format_data(hin.sample(3), hin.path_size)))