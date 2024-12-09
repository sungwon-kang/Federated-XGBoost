import math
import numpy as np
from sklearn.metrics import accuracy_score

import pyximport

pyximport.install()
from model.decision_tree import Tree
from model.loss_function import MultinomialDeviance

class BaseGradientBoosting():

    def __init__(self, loss, learning_rate, n_trees, max_depth,
                 outdim, min_samples_split=2):
        super().__init__()
        self.loss = loss
        self.learning_rate = learning_rate
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.trees = {}
        self.outdim = outdim
        self.cur_idx = -1


class GradientBoostingMultiClassifier(BaseGradientBoosting):
    def __init__(self, learning_rate, n_trees, max_depth, outdim, lamb, hes,
                 min_samples_split=2):
        super().__init__(MultinomialDeviance(outdim, lamb, hes), learning_rate, n_trees,
                         max_depth, outdim, min_samples_split)

    def print_nodes(self):
        for tree in self.trees:
            tree.print_nodes(tree.root_node, "")

    def print_leaf(self):
        for tree in self.trees:
            for i, leaf in enumerate(tree.leaf_nodes):
                print(f"{i}: {leaf.predict_value} {leaf.split_feature} {leaf.split_value}")

    def initialize_f_0(self, client_set):
        m = len(client_set)
        y_pred_set = []

        for i in range(m):
            X, _, _ = client_set[i]
            f_0 = np.zeros((X.shape[0], self.outdim), dtype=np.float64)
            y_pred_set.append(f_0)

        return y_pred_set

    def get_count_nodes_per_tree(self):
        counts=[]
        for tree in self.trees:
            count = tree.count_nodes(tree.root_node)
            counts.append([tree.leaf_count,tree.decision_count, count])
        return counts
    def get_tree_order(self, args, client_set):
        if args.joint == True:
            length = len(client_set)
            indices = np.arange(len(client_set))

            if args.method == "GinO":
                client_idx = np.array([np.random.permutation(indices) for _ in
                                           range(math.ceil(self.n_trees / length))], dtype=np.int32).reshape(-1)

            elif args.method == "GinM":
                client_idx = np.array([np.random.permutation(indices)[:math.ceil((self.max_depth - args.init_depth + 1) / args.levelUp)] for _ in
                                           range(self.n_trees)], dtype=np.int32)
                self.max_depth = args.init_depth

        else:
            client_idx = np.zeros(self.n_trees, dtype=np.int32)

        return client_idx

    def fit(self, client_set, test_set, args):

        client_idx = self.get_tree_order(args, client_set)

        self.trees = []
        ACCs = []
        for iter in range(0, self.n_trees):
            print('-----------------------------Learning of %d-th tree-----------------------------' % (iter + 1))
            if args.method == 'GinO':
                cur_client_idx = client_idx[iter]
            elif args.method == 'GinM':
                iter_client_idx = client_idx[iter]
                cur_client_idx = iter_client_idx[0]

            X, Y, _ = client_set[cur_client_idx]
            if iter > 0:
                y_pred = self.loss.update_f_m(X, self.trees, self.learning_rate, iter)
            else:
                y_pred = np.zeros((X.shape[0], self.outdim), dtype=np.float64)

            sub_index = np.random.permutation(X.shape[0])[:round(X.shape[0] * args.fraction)]
            self.trees.append(
                Tree(X[sub_index], Y[sub_index], y_pred[sub_index], self.max_depth, self.min_samples_split,
                     self.loss, cur_client_idx))

            if args.method == 'GinM':
                tree = self.trees[iter]
                for cur_client_idx in iter_client_idx[1:]:
                    tree.max_depth = tree.max_depth + args.levelUp
                    X, Y, _ = client_set[cur_client_idx]
                    tree.cur_idx = cur_client_idx

                    if iter > 0:
                        y_pred = self.loss.update_f_m(X, self.trees, self.learning_rate, iter)
                    else:
                        y_pred = np.zeros((X.shape[0], self.outdim), dtype=np.float64)

                    sub_index = np.random.permutation(X.shape[0])[:round(X.shape[0] * args.fraction)]

                    tree.leaf_nodes.clear()
                    tree.root_node = tree.grow(tree.root_node, X[sub_index], Y[sub_index], y_pred[sub_index])

                self.trees[iter] = tree
            
            if args.joint == True:
                self.adjust(client_set, iter)

            # 중간 트리 결과용
            testX, _, testY = test_set
            y_hat = self.predict(testX)
            acc = accuracy_score(testY, y_hat)
            ACCs.append(acc)

        counts = self.get_count_nodes_per_tree()
        np.savetxt(f'./results/{args.seed}_testacc_eachtree.txt', np.array(ACCs), delimiter=' ', fmt='%f')
        np.savetxt(f'./results/{args.seed}_all_nodes_eachtree.txt', np.array(counts), delimiter=' ', fmt='%f')
    
    def predict(self, X):
        y_pred = np.zeros((X.shape[0], self.outdim), dtype=np.float64)

        for tree in self.trees:
            y_pred += self.learning_rate * np.apply_along_axis(tree.root_node.get_predict_value, axis=1, arr=X)

        final_class = np.apply_along_axis(np.argmax, axis=1, arr=y_pred)
        return final_class

    def adjust(self, client_set, iter):

        for train in client_set:
            X, Y, _ = train

            if iter == 0:
                y_pred = np.zeros((X.shape[0], self.outdim), dtype=np.float64)
            else:
                y_pred = self.loss.update_f_m(X, self.trees, self.learning_rate, iter)

            self.trees[iter].root_node.calculate_gradients_inleaf(X, Y, y_pred)

        self.trees[iter].update_leafweight()
