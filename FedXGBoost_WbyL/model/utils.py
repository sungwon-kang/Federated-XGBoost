import sys
import random
import math
import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split

class util:

    def __init__(self, seed, n_clients=10, test_size=0.2):
        self.features = None
        self.encoder = None
        self.outdim = None
        self.seed = seed
        self.test_size = test_size
        self.n_clients = n_clients

    def get_datadst_setting(self, filename, args):
        train, test = self.getdata(filename)
        self.classes = np.unique(train[:, -1]).astype('int32')

        if args.joint == True:
            if args.env == "IID":
                client_set = self.split_data_for_IIDdst(train)
            elif args.env == "label_nonIID":
                client_set = self.split_data_for_nonIIDdst(train, args.alpha)
        else:
            client_set=[train.tolist()]

        client_set = self.fianl_processing(client_set)
        test_set = (test[:, :-1], self.encoder.transform(test[:, -1].reshape(-1, 1)), test[:, -1])

        return client_set, test_set

    def getdata(self, filename):

        filepath = f'./data/{filename}.csv'
        data = pd.read_csv(filepath).values.astype("float64")
        train, test = train_test_split(data, test_size=self.test_size,
                                       shuffle=True, stratify=data[:, -1], random_state=self.seed)

        self.encoder = self.transform_onehot(train[:, -1].reshape(-1, 1))
        self.outdim = len(np.unique(train[:, -1], axis=0))
        self.features = data.shape[1] - 1
        return train, test

    def fianl_processing(self, client_set):
        final_client_set = []
        for subset in client_set:
            subset = np.array(subset, dtype=np.float64)
            X, Y = subset[:, :-1], subset[:, -1]
            onehotY = self.encoder.transform(Y.reshape(-1, 1))
            final_client_set.append((X, onehotY, Y))
        return final_client_set

    def transform_onehot(self, rawY):
        encoder = OneHotEncoder(sparse=False, categories='auto')
        encoder.fit(rawY)
        return encoder

    def split_dataset_by_class(self, train):

        class_set = []
        for i in range(len(self.classes)):
            i_class_set = train[train[:, -1] == i, :]
            class_set.append(i_class_set)
        return class_set

    def split_data_for_IIDdst(self, train):

        class_set = self.split_dataset_by_class(train)

        client_train = [[] for _ in range(self.n_clients)]
        for subset in class_set:
            length = subset.shape[0] // self.n_clients
            start = 0
            for i in range(self.n_clients):

                end = length + start
                if i == (self.n_clients - 1):
                    part_t = subset[start:]
                else:
                    part_t = subset[start:end]

                client_train[i] += part_t.tolist()
                start = end

        return client_train

    def split_data_for_nonIIDdst(self, train, alpha):

        y_train = train[:, -1]

        num = 2
        if self.outdim / self.n_clients > 2.0:
            num = math.ceil(self.outdim / self.n_clients) * alpha

        times = [0 for _ in range(self.outdim)]
        temp = np.random.permutation(self.outdim)
        k = 0

        contain = []
        for i in range(self.n_clients):
            current = []
            for j in range(num):
                while True:
                    index = temp[k]
                    if index not in current:
                        current.append(index)
                        times[index] += 1
                        k += 1
                        if k == self.outdim:
                            temp = np.random.permutation(self.outdim)
                            k = 0
                        break
                    k += 1
            contain.append(current)


        splited_index = []
        for i in range(self.outdim):
            idx_k = np.where(y_train == i)[0]
            np.random.shuffle(idx_k)
            split = np.array_split(idx_k, times[i])
            splited_index.append(split)

        # debug = np.zeros((self.n_clients, self.outdim))
        client_train = [[] for _ in range(self.n_clients)]
        for i, current in enumerate(contain):
            for c in current:
                times[c] -= 1
                class_data = splited_index[c]
                data = class_data[times[c]]
                # debug[i][c] += len(data)
                client_train[i]+=train[data].tolist()

        return client_train

