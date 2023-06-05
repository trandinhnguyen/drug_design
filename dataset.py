import os
from qsprpred.data.data import QSPRDataset
from qsprpred.data.utils.descriptorsets import FingerprintSet
from qsprpred.data.utils.descriptorcalculator import DescriptorsCalculator
from qsprpred.data.utils.datasplitters import scaffoldsplit


def create_dataset(
    dataset,
    dataset_name,
    target_prop,
    th=[6.5],
    test_size=0.2,
    path="data/datasets/qsar",
    save=False,
):
    """
    Params:
        dataset: a DataFrame
        target_prop: Column name in dataset
        th: threshold of classification task
        path: string. Then path = path + dataset_name
    Returns:
        QSPRDataset
    """
    path = os.path.join(path, dataset_name)
    os.makedirs(path, exist_ok=True)

    # create the data set
    dataset = QSPRDataset(
        name=dataset_name,
        df=dataset,
        target_prop=target_prop,
        store_dir=path,
    )
    dataset.makeClassification(th=th)

    # Calculate MorganFP and physicochemical properties
    feature_calculator = DescriptorsCalculator(
        descsets=[FingerprintSet(fingerprint_type="MorganFP", radius=3, nBits=2048)]
    )

    # split on scaffolds
    split = scaffoldsplit(test_fraction=test_size)

    dataset.prepareDataset(split=split, feature_calculator=feature_calculator)

    if save:
        dataset.save()

    print(f"Number of samples train set: {len(dataset.y)}")
    print(
        f"Number of samples test set: {len(dataset.y_ind)}, {len(dataset.y_ind) / len(dataset.df) * 100}%"
    )
    return dataset
