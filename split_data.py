import pandas as pd
import torch


class DataManager:

    def __init__(self, data, labels, folds):

        self.raw_data = data
        self.data = self.chunks(data, folds)
        self.labels = self.chunks(labels, folds)
        self.folds = folds
        self.index = 0
    
    def chunks(self, seq, n):

        avg = len(seq) / float(n)
        out = []
        last = 0.0
        while last < len(seq):
            out.append(seq[int(last):int(last + avg)])
            last += avg
        return out

    def getNext(self):
        data_copy = self.data.copy()
        labels_copy = self.labels.copy()

        data_test, labels_test = data_copy.pop(self.index), labels_copy.pop(self.index)
        test = self.pre_process(data_test, labels_test)

        temp_train = list()
        temp_labels = list()
        for l in data_copy:
            temp_train += l

        for s in labels_copy:
            temp_labels += s

        train = self.pre_process(temp_train, temp_labels)
        self.index += 1

        return train, test

    def pre_process(self, data):
        raise NotImplementedError
        return None



