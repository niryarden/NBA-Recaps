from datasets import load_dataset
from config import config
import os
import json
import pandas as pd

DEFAULT_VALUE = None


def load_results_from_files():
    game_data = []
    folder_path = "recaps/"
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            with open(file_path) as f:
                data = json.load(f)
                curr_metadata = data["metadata"]
                game_id = curr_metadata["game_id"]

                players_names = (
                        curr_metadata['home_team_players_names'] +
                        curr_metadata['home_team_names_I'] +
                        curr_metadata['away_team_players_names'] +
                        curr_metadata['away_team_names_I']
                )

                game_data.append({
                    'game_id': game_id,
                    'reference_recap': data["reference_recap"],
                    'generated_recap': data["generated_recap"],
                    'player_names': players_names,
                    'metadata': curr_metadata
                })

    return pd.DataFrame(game_data)


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
