from datasets import load_dataset
from config import config


class PredictedGame:
    def __init__(self, game_id, reference, prediction, player_names):
        self.game_id = game_id
        self.reference = reference
        self.prediction = prediction
        self.player_names = player_names


def load_nba_dataset(dataset=config["dataset_name"], limit=None):
    raw_datasets = load_dataset(dataset, trust_remote_code=True)

    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["validation"]
    test_dataset = raw_datasets["test"]
    unsupervised_dataset = raw_datasets["unsupervised"]

    if limit:
        train_dataset = train_dataset.select(range(limit))
        eval_dataset = eval_dataset.select(range(limit))
        test_dataset = test_dataset.select(range(limit))
        unsupervised_dataset = unsupervised_dataset.select(range(limit))

    return train_dataset, eval_dataset, test_dataset, unsupervised_dataset
