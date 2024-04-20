import json
import os

import pandas
from torch.utils.data import Dataset

from src.dataset.input_preprocessor import InputPreprocessor


class Text2SqlDataset(Dataset):
    def __init__(self, args, split):
        super().__init__()
        self.input_preprocessor = InputPreprocessor(args)
        df = pandas.read_csv(args.data_path)
        nat_sql = json.load(open(args.nat_sql_path))

        # split into train and val datasets in a way that no database is shared
        total_example_count = df.shape[0]
        train_example_count = total_example_count * 0.9
        unique_db_with_example_counts = df.groupby("spider_db_name").count()
        count = 0
        for i in range(unique_db_with_example_counts.shape[0]):
            count += unique_db_with_example_counts['question'].iloc[i]
            if count >= train_example_count:
                break

        df = df.head(count) if split == "train" else df.tail(df.shape[0] - count)
        nat_sql = nat_sql[:count] if split == "train" else nat_sql[count:]

        self.dataset = []
        for i in range(df.shape[0]):
            self.dataset.append({
                "input": self.input_preprocessor.get_input(df['question'].iloc[i], df['schema'].iloc[i]),
                "label": self.input_preprocessor.get_label(nat_sql[i]['NatSQL'])
            })

    def __getitem__(self, item):
        example = self.dataset[item]
        return example["input"], example["label"]

    def __len__(self):
        return len(self.dataset)
