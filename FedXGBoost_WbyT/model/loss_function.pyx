import numpy as np
cimport numpy as np
np.import_array()

ctypedef double DTYPE_t
from cython.parallel import prange
import sys

cdef softmax(np.ndarray[DTYPE_t, ndim=2] x):
    cdef np.ndarray[DTYPE_t, ndim=2] exp_x
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

cdef class MultinomialDeviance:
    cdef int outdim
    cdef double lamb
    cdef int hes

    def __init__(self, int outdim, double lamb, bint hes):
        self.outdim = outdim
        self.lamb = lamb
        self.hes = hes

    def update_f_m(self,
                   np.ndarray[DTYPE_t, ndim=2] data,
                   object tree,
                   double learning_rate,
                   int cur_iter):

        cdef np.ndarray[DTYPE_t, ndim=2] pred
        pred = learning_rate * np.apply_along_axis(
                               tree[0].root_node.get_predict_value,
                               axis=1,
                               arr=data)

        for iter in range(1, cur_iter):
            pred += learning_rate * \
                    np.apply_along_axis(
                    tree[iter].root_node.get_predict_value,
                    axis=1,
                    arr=data)

        return pred

    def get_grad_hessian(self,
                         np.ndarray[DTYPE_t, ndim=2] y_true,
                         np.ndarray[DTYPE_t, ndim=2] y_prob):

        cdef int m = y_true.shape[0]
        cdef np.ndarray[DTYPE_t, ndim=2] y_pred = softmax(y_prob)
        cdef np.ndarray[DTYPE_t, ndim=2] g = y_pred - y_true
        cdef np.ndarray[DTYPE_t, ndim=3] H = np.zeros((m, self.outdim, self.outdim), dtype=np.double)

        cdef Py_ssize_t i, j
        if self.hes == 1:
            for i in prange(m, nogil=True):
                for j in range(self.outdim):
                    H[i, j, j] = 1.0 * (y_pred[i, j] * (1.0 - y_pred[i, j]))

        return g, H

    def getInverseMat(self,
                      np.ndarray[DTYPE_t, ndim=2] H):
        cdef np.ndarray[DTYPE_t, ndim=2] lamb_matrix
        try:
            lamb_matrix = np.eye(self.outdim) * self.lamb
            inverse_matrix = np.linalg.inv(lamb_matrix + H)
        except np.linalg.LinAlgError:
            print("역행렬이 존재하지 않습니다.")
            return None

        return inverse_matrix

    def get_leaf_weight(self,
                        np.ndarray[DTYPE_t, ndim=2] y_true,
                        np.ndarray[DTYPE_t, ndim=2] y_prob):

        cdef np.ndarray[DTYPE_t, ndim=2] g
        cdef np.ndarray[DTYPE_t, ndim=3] H
        g, H = self.get_grad_hessian(y_true, y_prob)

        cdef np.ndarray[DTYPE_t, ndim=1] G = np.sum(g, axis=0)
        cdef np.ndarray[DTYPE_t, ndim=2] H_sum = np.sum(H, axis=0)
        cdef np.ndarray[DTYPE_t, ndim=2] inverse_matrix = self.getInverseMat(H_sum)

        return inverse_matrix @ G

    def update_leaf_values(self,
                           np.ndarray[DTYPE_t, ndim=2] y_true,
                           np.ndarray[DTYPE_t, ndim=2] y_prob):
        cdef np.ndarray[DTYPE_t, ndim=1] leaf_weight
        leaf_weight = -1.0 * self.get_leaf_weight(y_true, y_prob)
        return leaf_weight

    def getGain(self,
                np.ndarray[DTYPE_t, ndim=2] y_true,
                np.ndarray[DTYPE_t, ndim=2] y_prob):

        cdef np.ndarray[DTYPE_t, ndim=2] g
        cdef np.ndarray[DTYPE_t, ndim=3] H
        g, H = self.get_grad_hessian(y_true, y_prob)
        cdef np.ndarray[DTYPE_t, ndim=1] G = np.sum(g, axis=0)
        cdef np.ndarray[DTYPE_t, ndim=2] H_sum = np.sum(H, axis=0)
        cdef np.ndarray[DTYPE_t, ndim=2] inverse_matrix = self.getInverseMat(H_sum)

        cdef double gain = 0.5 * (G.T @ (inverse_matrix @ G))

        return gain

    def find_best_split(self, tuple args):
        cdef:
            np.ndarray[int, ndim=1] features = args[0]
            np.ndarray[DTYPE_t, ndim=2] data = args[1]
            np.ndarray[DTYPE_t, ndim=2] target = args[2]
            np.ndarray[DTYPE_t, ndim=2] label = args[3]
            double now_gain = args[4]

            double gain = -1.0 * np.inf
            int split_feature = -1
            double split_value = -1.0 * np.inf
            np.ndarray[np.uint8_t, ndim=1, cast=True] left_index_of_now_data = None
            np.ndarray[np.uint8_t, ndim=1, cast=True] right_index_of_now_data = None

        for feature in features:
            feature_values = np.unique(data[:, feature])
            sorted_values = np.sort(feature_values)
            for fea_val in sorted_values:
                left_index = data[:, feature] < fea_val
                right_index = ~left_index

                left_target = target[left_index]
                left_label = label[left_index]

                right_target = target[right_index]
                right_label = label[right_index]


                if np.any(left_index) == True:
                    left_gain = self.getGain(left_label, left_target)
                else:
                    left_gain = 0

                if np.any(right_index) == True:
                    right_gain = self.getGain(right_label, right_target)
                else:
                    right_gain = 0

                sum_gain = (left_gain + right_gain) - now_gain  # - gamma
                print(f'------feature: %d, Split value: %.3f, Left node gain: %.3f, Right node gain: %.3f, Total gain: %.3f' %
                      (feature, fea_val, left_gain, right_gain, sum_gain))

                if gain is None or sum_gain > gain:
                    split_feature = feature
                    split_value = fea_val
                    gain = sum_gain
                    left_index_of_now_data = left_index
                    right_index_of_now_data = right_index

        return (split_feature, split_value, gain, left_index_of_now_data, right_index_of_now_data)