from split_data import DataManager

class KFold():

    def __init__(self, data, labels, num_folds):
        self.data = DataManager(data, labels, num_folds)
        self.folds = num_folds
        self.accuracy = []
        self.loss = []
        self.current_model = None

    def run(self):

        for k in range(self.folds):
            train, test = self.data.getNext()
            accuracy, loss = self.train(train, test)
            self.accuracy.append(accuracy)
            self.loss.append(loss)

    def getAccuracy(self):
        return (sum(self.accuracy)/len(self.accuracy))


    def getLoss(self):
        return (sum(self.loss)/len(self.loss))

    def train(self):
        raise NotImplementedError
        return None

    def test(self):
        raise NotImplementedError
        return None

