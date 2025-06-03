from dataset.DatasetBucket import DatasetBucket
from dataset.DatasetsIdentifier import DatasetIdentifier


class DatasetBucketFactory():

    @staticmethod
    def create(dataset_identifier: DatasetIdentifier) -> DatasetBucket:
        if dataset_identifier == DatasetIdentifier.TWINPROMPT:
            dataset_bucket = DatasetBucket("twinprompt.json")
        elif dataset_identifier == DatasetIdentifier.ABLATION_NON_SIMILAR:
            dataset_bucket = DatasetBucket("ablation_non_similarity.json")
        elif dataset_identifier == DatasetIdentifier.STRONGREJECT:
            dataset_bucket = DatasetBucket("strongreject.json")
        elif dataset_identifier == DatasetIdentifier.HARMBENCH_VALIDATION:
            dataset_bucket = DatasetBucket("harmbench_validation.json")
        elif dataset_identifier == DatasetIdentifier.JAILBREAKBENCH:
            dataset_bucket = DatasetBucket("jailbreakbench.json")
        elif dataset_identifier == DatasetIdentifier.ADVBENCH:
            dataset_bucket = DatasetBucket("advbench.json")
        else:
            raise Exception(f"Unknown dataset identifier: {dataset_identifier}")
        return dataset_bucket
