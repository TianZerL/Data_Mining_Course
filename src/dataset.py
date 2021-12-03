import pandas as pd


from sklearn import datasets, preprocessing, utils


class DataSet(object):
    def __init__(self) -> None:
        self.org_dataset = self.get_dataset()


# 连续实数
class Iris(DataSet):
    def __init__(self) -> None:
        super().__init__()
        self.data = self.org_dataset.data
        self.label = self.org_dataset.target

    def get_dataset(self):
        return datasets.load_iris()


# 连续实数、整数
class Wine(DataSet):
    def __init__(self) -> None:
        super().__init__()
        self.data = self.org_dataset.data
        self.label = self.org_dataset.target

    def get_dataset(self):
        return datasets.load_wine()


# 归一化
class WineNormalized(DataSet):
    def __init__(self) -> None:
        super().__init__()
        self.data = preprocessing.MinMaxScaler().fit_transform(self.org_dataset.data)
        self.label = self.org_dataset.target

    def get_dataset(self):
        return datasets.load_wine()


# 类别
class Car(DataSet):
    def __init__(self) -> None:
        super().__init__()
        self.data = pd.get_dummies(self.org_dataset[range(0, 6)])
        self.label = self.org_dataset[6]

    def get_dataset(self):
        return pd.read_table(
            r"https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data",
            sep=",",
            header=None,
        )


# 二进制二分类
class Heart(DataSet):
    def __init__(self) -> None:
        super().__init__()
        self.data = self.org_dataset[range(1, 23)]
        self.label = self.org_dataset[0]

    def get_dataset(self):
        return pd.read_table(
            r"https://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECT.train",
            sep=",",
            header=None,
        )


# 二进制二分类
class HeartMore(DataSet):
    def __init__(self) -> None:
        super().__init__()
        self.data = self.org_dataset[range(1, 23)]
        self.label = self.org_dataset[0]

    def get_dataset(self):
        return utils.shuffle(
            pd.concat(
                [
                    pd.read_table(
                        r"https://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECT.train",
                        sep=",",
                        header=None,
                    ),
                    pd.read_table(
                        r"https://archive.ics.uci.edu/ml/machine-learning-databases/spect/SPECT.test",
                        sep=",",
                        header=None,
                    ),
                ]
            )
        )


# 类别, OneHotEncode
class MushroomOneHotEncode(DataSet):
    def __init__(self) -> None:
        super().__init__()
        self.data = pd.get_dummies(self.org_dataset[range(1, 23)])
        self.label = pd.Series(self.org_dataset[0])

    def get_dataset(self):
        return pd.read_table(
            r"https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data",
            sep=",",
            header=None,
        )


# 类别
class Mushroom(DataSet):
    def __init__(self) -> None:
        super().__init__()
        self.data = pd.DataFrame.apply(
            self.org_dataset[range(1, 23)], preprocessing.LabelEncoder().fit_transform
        )
        self.label = pd.Series(self.org_dataset[0])

    def get_dataset(self):
        return pd.read_table(
            r"https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data",
            sep=",",
            header=None,
        )
