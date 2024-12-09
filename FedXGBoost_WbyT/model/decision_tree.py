from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np
import math

from colorama import init as colorama_init
from colorama import Fore
from colorama import Style

class Node:
    def __init__(self, outdim, idx, split_feature=None, split_value=None, is_leaf=False, loss=None,
                 deep=None):
        self.loss = loss
        self.split_feature = split_feature
        self.split_value = split_value

        # only adjust
        self.G_sum = np.zeros((outdim))
        self.H_sum = np.zeros((outdim, outdim))

        self.is_leaf = is_leaf
        self.predict_value = None
        self.left_child = None
        self.right_child = None
        self.deep = deep
        self.cur_idx = idx

    def update_predict_value(self, y, targets):
        self.predict_value = self.loss.update_leaf_values(y, targets)

    def self_update_predict_value(self):
        inverse_matrix = self.loss.getInverseMat(self.H_sum)
        self.predict_value = -1.0 * (inverse_matrix @ self.G_sum)

    def get_predict_value(self, instance):
        if self.is_leaf:
            return self.predict_value
        if instance[self.split_feature] < self.split_value:
            return self.left_child.get_predict_value(instance)
        else:
            return self.right_child.get_predict_value(instance)

    def recursive_update_leaf(self, node):
        if node.is_leaf == True:
            node.self_update_predict_value()
            return

        self.recursive_update_leaf(node.left_child)
        self.recursive_update_leaf(node.right_child)

    def calculate_gradients_inleaf(self, instance, y_true, y_pred):
        if self.is_leaf:
            g, H = self.loss.get_grad_hessian(y_true, y_pred)
            G_sum = np.sum(g, axis=0)
            H_sum = np.sum(H, axis=0)
            self.G_sum += G_sum
            self.H_sum += H_sum
            return

        left_index = instance[:, self.split_feature] < self.split_value
        self.left_child.calculate_gradients_inleaf(instance[left_index],
                                                   y_true[left_index],
                                                   y_pred[left_index])

        right_index = ~left_index
        self.right_child.calculate_gradients_inleaf(instance[right_index],
                                                    y_true[right_index],
                                                    y_pred[right_index])


class Tree:
    def __init__(self, X, Y, target, max_depth, min_samples_split, loss, cur_client_idx, max_workers=20):
        self.loss = loss
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.leaf_nodes = []
        self.max_workers = max_workers
        self.cur_idx = cur_client_idx # debugging var
        self.root_node = self.build_tree(X, Y, target, depth=0)

        self.leaf_count=0
        self.decision_count=0
    def update_leafweight(self):
        for leaf in self.leaf_nodes:
            leaf.predict_value = None
            inverse_matrix = self.loss.getInverseMat(leaf.H_sum)
            leaf.predict_value = -1.0 * (inverse_matrix @ leaf.G_sum)

    def update_rec_leaf(self, root):
        root.recursive_update_leaf(root)

    def clear_leafnodes(self):
        for node in self.leaf_nodes:
            node.predict_value = None

    def count_nodes(self, node):
        if node is None:
            return 0

        if node.is_leaf:
            self.leaf_count+=1
        else:
            self.decision_count+=1

        return 1 + self.count_nodes(node.left_child) + self.count_nodes(node.right_child)

    def build_tree(self, data, label, target, depth=0):

        n_label = len(np.unique(label, axis=0))
        if depth < self.max_depth \
                and data.shape[0] >= self.min_samples_split \
                and n_label > 1:
            best_split = pd.Series({'gain': None,
                                    'split_feature': None,
                                    'split_value': None,
                                    'left_index': None,
                                    'right_index': None})

            now_gain = self.loss.getGain(label, target)
            print('--Tree depth: %d' % depth)

            feature_range = np.arange(data.shape[1])
            if len(feature_range) <= self.max_workers:
                subsets = [[i] for i in feature_range]
            else:
                values_counts = np.array([[feature, len(np.unique(data[:, feature]))] for feature in feature_range])
                sorted_values = values_counts[values_counts[:, 1].argsort()]

                sample_size = math.ceil(sorted_values.shape[0] / self.max_workers)
                sets = [sorted_values[i * sample_size: (i + 1) * sample_size, :] for i in range(self.max_workers)]

                subsets = [[] for _ in range(self.max_workers)]

                for i in range(self.max_workers):
                    for subset in sets:
                        sample_size = math.ceil(len(subset) / self.max_workers)
                        subsets[i] += subset[i * sample_size: (i + 1) * sample_size, 0].tolist()

            data_for_fit = [(np.array(subset, dtype=np.int32), data, target, label, now_gain) for subset in subsets if
                            len(subset) > 0]

            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                best_candidates = list(executor.map(self.loss.find_best_split, data_for_fit))

            for candidate in best_candidates:
                if best_split['gain'] is None or candidate[2] > best_split['gain']:
                    best_split['split_feature'] = candidate[0]
                    best_split['split_value'] = candidate[1]
                    best_split['gain'] = candidate[2]
                    best_split['left_index'] = candidate[3]
                    best_split['right_index'] = candidate[4]

            print('--Best split feature:', best_split['split_feature'])
            print('--Best split value:', best_split['split_value'])
            print('--Best gain:', best_split['gain'])

            if best_split['gain'] > 0:
                node = Node(label.shape[1], self.cur_idx, best_split['split_feature'], best_split['split_value'], deep=depth)

                node.left_child = self.build_tree(data[best_split['left_index']], label[best_split['left_index']],
                                                  target[best_split['left_index']], depth + 1)
                node.right_child = self.build_tree(data[best_split['right_index']], label[best_split['right_index']],
                                                   target[best_split['right_index']], depth + 1)
                return node

        node = Node(label.shape[1], idx=self.cur_idx, is_leaf=True, loss=self.loss, deep=depth)
        node.update_predict_value(label, target)
        self.leaf_nodes.append(node)
        return node

    def grow(self, node, X, Y, y_pred):
        n_label = len(np.unique(Y, axis=0))
        if node.is_leaf == True and (node.deep >= self.max_depth
                                     or n_label <= 1
                                     or X.shape[0] < self.min_samples_split):
            self.leaf_nodes.append(node)
            return node

        if node.is_leaf == True:
            node = self.build_tree(X, Y, y_pred, node.deep)

        else:
            feature = node.split_feature
            fea_val = node.split_value

            left_index = X[:, feature] < fea_val
            right_index = ~left_index

            node.left_child = self.grow(node.left_child, X[left_index], Y[left_index], y_pred[left_index])
            node.right_child = self.grow(node.right_child, X[right_index], Y[right_index], y_pred[right_index])

        return node
