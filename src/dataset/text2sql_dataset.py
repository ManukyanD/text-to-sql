import json

import pandas
from torch.utils.data import Dataset

from src.dataset.input_preprocessor import InputPreprocessor


class Text2SqlDataset(Dataset):
    def __init__(self, args, split):
        super().__init__()
        self.input_preprocessor = InputPreprocessor(args)
        df = pandas.read_csv(args.data_path)
        nat_sql = json.load(open(args.nat_sql_path))
        df.insert(0, 'nat_sql', [item["NatSQL"] for item in nat_sql])

        # split into train and val datasets in a way that no database is shared
        total_example_count = df.shape[0]
        train_example_count = total_example_count * 0.9
        unique_db_with_example_counts = df.groupby("spider_db_name").count()
        train_db_names = []
        count = 0
        for i, row in unique_db_with_example_counts.iterrows():
            count += row.question
            train_db_names.append(row.name)
            if count >= train_example_count:
                break
        df = df[df.spider_db_name.isin(train_db_names) if split == "train" else ~df.spider_db_name.isin(train_db_names)]

        self.dataset = []
        for i, row in df.iterrows():
            self.dataset.append({
                "input": self.input_preprocessor.get_input(row.question, row.schema),
                "label": self.input_preprocessor.get_label(row.nat_sql)
            })

    def __getitem__(self, item):
        example = self.dataset[item]
        return example["input"], example["label"]

    def __len__(self):
        return len(self.dataset)
