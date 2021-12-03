from sklearn import model_selection

# 采用K折交叉验证与F1分数进行模型评估
def classify_skf_f1_score(dataset, classifier, cv=10):
    return model_selection.cross_val_score(
        estimator=classifier,
        X=dataset.data,
        y=dataset.label,
        scoring="f1_weighted",
        cv=cv,
    ).mean()
