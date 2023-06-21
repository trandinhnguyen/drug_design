from qsprpred.models.models import QSPRsklearn


def train(dataset, params, classifier, name, base_dir="data/models/qsar/"):
    model = QSPRsklearn(
        name=name,
        base_dir=base_dir,
        data=dataset,
        alg=classifier,
    )

    model.gridSearch(search_space_gs=params)
    model.evaluate()
    model.fit()
    return model
