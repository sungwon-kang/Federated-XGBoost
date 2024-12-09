from concurrent.futures import ProcessPoolExecutor
import pandas as pd
import numpy as np
import math
import sys

import multiprocessing


class Node:
    def __init__(self, outdim, split_feature=None, split_value=None, is_leaf=False, loss=None,
                 deep=None):
        self.loss = loss
        self.split_feature = split_feature
        self.split_value = split_value
        self.outdim = outdim
        # only adjust
        self.G_sum = np.zeros((outdim))
        self.H_sum = np.zeros((outdim, outdim))

        self.is_leaf = is_leaf
        self.predict_value = None
        self.left_child = None
        self.right_child = None
        self.deep = deep

    def update_predict_value(self, y, targets, lr):
        self.predict_value = self.loss.update_leaf_values(y, targets) * lr

    def get_predict_value(self, instance, predict_value=None):
        if predict_value is None:
            predict_value = np.zeros((self.outdim))

        if self.is_leaf:
            return predict_value + self.predict_value
        if instance[self.split_feature] < self.split_value:
            return self.left_child.get_predict_value(instance, predict_value + self.predict_value)
        else:
            return self.right_child.get_predict_value(instance, predict_value + self.predict_value)

    def calculate_gradients(self, client_set, lr):

        for _, y_true, y_pred in client_set:
            g, H = self.loss.get_grad_hessian(y_true, y_pred)
            G_sum = np.sum(g, axis=0)
            H_sum = np.sum(H, axis=0)
            self.G_sum += G_sum
            self.H_sum += H_sum

        inv_mat = self.loss.getInverseMat(self.H_sum)
        self.predict_value = (-1.0 * inv_mat @ self.G_sum) * lr

        if self.is_leaf:
            return

        left_index = [instance[:, self.split_feature] < self.split_value for instance, _, _ in client_set]
        left_client_set = []
        right_client_set = []

        for subset, left_idx in zip(client_set, left_index):
            instance = subset[0]
            y_true = subset[1]
            y_pred = subset[2]
            y_pred += self.predict_value

            left_instance = instance[left_idx]
            left_y_true = y_true[left_idx]
            left_y_pred = y_pred[left_idx]
            left_client_set.append((left_instance, left_y_true, left_y_pred))

            right_index = ~left_idx
            right_instance = instance[right_index]
            right_y_true = y_true[right_index]
            right_y_pred = y_pred[right_index]
            right_client_set.append((right_instance, right_y_true, right_y_pred))

        self.left_child.calculate_gradients(left_client_set, lr)
        self.right_child.calculate_gradients(right_client_set, lr)


class Tree:
    def __init__(self, X, Y, target, max_depth, min_samples_split, loss, lr, max_workers=24):
        self.loss = loss
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.nodes = []
        self.lr=lr
        self.max_workers = max_workers
        self.root_node = self.build_tree(X, Y, target, depth=0)

        self.leaf_count = 0
        self.decision_count = 0

    def count_nodes(self, node):
        if node is None:
            return 0

        if node.is_leaf:
            self.leaf_count+=1
        else:
            self.decision_count+=1

        return 1 + self.count_nodes(node.left_child) + self.count_nodes(node.right_child)

    def clear_nodes_debug(self):
        for node in self.nodes:
            node.predict_value = None



    def print_tree(self, node, prefix=""):
        if node.is_leaf:
            print(prefix + "+- <ExNode>")
            print(prefix + "=> ", node.predict_value)
            return

        else:
            left = node.left_child
            right = node.right_child
            print(prefix + "+- <InNode>")
            print(prefix + "=> ", node.split_feature, node.split_value)
            print(prefix + "=> ", node.predict_value)
            self.print_tree(right, prefix + "| \t")
            self.print_tree(left, prefix + "| \t")

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

            node = Node(label.shape[1], best_split['split_feature'], best_split['split_value']
                        ,loss=self.loss, deep=depth)
            node.update_predict_value(label, target, self.lr)
            if best_split['gain'] > 0:
                target += node.predict_value

                node.left_child = self.build_tree(data[best_split['left_index']], label[best_split['left_index']],
                                                  target[best_split['left_index']], depth + 1)
                node.right_child = self.build_tree(data[best_split['right_index']], label[best_split['right_index']],
                                                   target[best_split['right_index']], depth + 1)
                self.nodes.append(node)
                return node

        node = Node(label.shape[1], is_leaf=True, loss=self.loss, deep=depth)
        node.update_predict_value(label, target, self.lr)
        self.nodes.append(node)
        return node

    def grow(self, node, X, Y, y_pred):
        self.nodes.append(node)
        n_label = len(np.unique(Y, axis=0))
        if node.is_leaf == True and (node.deep >= self.max_depth
                                     or n_label <= 1
                                     or X.shape[0] < self.min_samples_split):

            return node

        if node.is_leaf == True:
            node = self.build_tree(X, Y, y_pred + node.predict_value, node.deep)

        else:
            feature = node.split_feature
            fea_val = node.split_value

            left_index = X[:, feature] < fea_val
            right_index = ~left_index

            y_pred += node.predict_value
            node.left_child = self.grow(node.left_child, X[left_index], Y[left_index], y_pred[left_index])
            node.right_child = self.grow(node.right_child, X[right_index], Y[right_index], y_pred[right_index])

        return node
