# 分类算法：朴素贝叶斯 决策树 支持向量机
from sklearn import naive_bayes, tree, neural_network

from classfifcation import classify_skf_f1_score


def cal_scores(dataset):
    return (
        classify_skf_f1_score(dataset, naive_bayes.GaussianNB()),
        classify_skf_f1_score(dataset, tree.DecisionTreeClassifier()),
        classify_skf_f1_score(
            dataset,
            neural_network.MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=3000),
        ),
    )


def print_scores(dataset_text, scores):
    print(
        dataset_text + "数据集"
        "\n--------------------\n"
        "高斯朴素贝叶斯:\t\t%f\n"
        "决策树CART:\t\t%f\n"
        "BP神经网络:\t\t%f\n"
        "\n--------------------\n" % scores
    )
