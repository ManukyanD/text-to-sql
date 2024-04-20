import argparse
import os

import pandas
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.dataset.input_preprocessor import InputPreprocessor
from src.dataset.output_postprocessor import OutputPostprocessor
from src.device import to_device
from src.tokenizer import get_tokenize_fn


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='ManukyanD/flan-t5-for-text2sql',
                        help='Model name (default: "ManukyanD/flan-t5-for-text2sql"')
    parser.add_argument('--data-path', type=str, required=True,
                        help='Test data path.'),
    parser.add_argument('--prefix', type=str, default="generate SQL",
                        help='Task prefix to add before input (default: "generate SQL'),
    parser.add_argument('--max-source-length', type=int, default=512,
                        help='The maximum total input sequence length after tokenization (default: 512).')
    parser.add_argument('--max-output-length', type=int, default=512,
                        help='The maximum total output number of tokens (default: 512).')
    parser.add_argument('--tables-path', type=str, default=os.path.join(".", "data", "tables_for_natsql.json"),
                        help='Path to tables json (default: "./data/tables_for_natsql.json").')
    parser.add_argument('--output-path', type=str, default=os.path.join(".", "pred.txt"),
                        help='Path to file where to write the output (default: "./pred.txt").')

    return parser.parse_args()


class InferDataset(Dataset):
    def __init__(self, args):
        self.input_preprocessor = InputPreprocessor(args)
        self.df = pandas.read_csv(args.data_path)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        return self.input_preprocessor.get_input(self.df.question.iloc[item], self.df.schema.iloc[item]), \
            self.df.spider_db_name.iloc[item]


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model.eval()
    to_device(model)

    tokenize_source = get_tokenize_fn(tokenizer, args.max_source_length)
    output_postprocessor = OutputPostprocessor(args)

    dataset = InferDataset(args)
    data_loader = DataLoader(dataset, batch_size=32)
    predictions = []
    for batch in data_loader:
        inputs, db_names = batch
        inputs = to_device(tokenize_source(inputs))
        input_ids = inputs.input_ids
        attention_mask = inputs.attention_mask
        with torch.no_grad():
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                     max_length=args.max_output_length)
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for index, output in enumerate(outputs):
            output = output_postprocessor.get_output(output, db_names[index])
            predictions.append(output)
    with open(args.output_path, 'w') as f:
        for prediction in predictions:
            f.write(f"{prediction}\n")


if __name__ == '__main__':
    main()
