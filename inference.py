import argparse
import os

import pandas
import torch
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


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model)
    model.eval()
    to_device(model)

    tokenize_source = get_tokenize_fn(tokenizer, args.max_source_length)
    input_preprocessor = InputPreprocessor(args)
    output_postprocessor = OutputPostprocessor(args)

    df = pandas.read_csv(args.data_path)
    predictions = []
    for i in range(df.shape[0]):
        input = input_preprocessor.get_input(df['question'].iloc[i], df['schema'].iloc[i])
        input_ids = to_device(tokenize_source(input).input_ids)
        with torch.no_grad():
            output = model.generate(input_ids=input_ids, max_length=args.max_output_length)
        output = tokenizer.decode(output[0], skip_special_tokens=True)
        output = output_postprocessor.get_output(output, df["spider_db_name"].iloc[i])
        predictions.append(output)
    with open(args.output_path, 'w') as f:
        for prediction in predictions:
            f.write(f"{prediction}\n")


if __name__ == '__main__':
    main()
