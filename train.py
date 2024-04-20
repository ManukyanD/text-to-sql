import argparse
import os

import torch
from tokenizers import AddedToken
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup

from src.dataset.text2sql_dataset import Text2SqlDataset
from src.device import to_device
from src.tokenizer import get_tokenize_fn


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--checkpoints-dir', type=str, default=os.path.join('.', 'checkpoints'),
                        help='The directory to save checkpoints to (default: "./checkpoints").')
    parser.add_argument('--prefix', type=str, default="generate SQL",
                        help='Task prefix to add before input (default: "generate SQL'),
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs (default: 10).')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Base learning rate (default: 1e-4).')
    parser.add_argument('--weight-decay', type=float, default=0.01,
                        help='Weight decay (default: 0.01).')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size (default: 4).')
    parser.add_argument('--max-source-length', type=int, default=512,
                        help='The maximum total input sequence length after tokenization (default: 512).')
    parser.add_argument('--max-target-length', type=int, default=512,
                        help='The maximum total target sequence length after tokenization (default: 512).')
    parser.add_argument('--data-path', type=str, default=os.path.join(".", "data", "text-to-sql_from_spider.csv"),
                        help='Train data path (default: "./data/text-to-sql_from_spider.csv"'),
    parser.add_argument('--nat-sql-path', type=str, default=os.path.join(".", "data", "train_spider-natsql.json"),
                        help='Path to json with NatSQL (default: "./data/train_spider-natsql.json"'),

    return parser.parse_args()


def fit(args, model, tokenizer, train_loader, test_loader, optimizer, scheduler):
    running_loss = 0
    tokenize_source = get_tokenize_fn(tokenizer, args.max_source_length)
    tokenize_target = get_tokenize_fn(tokenizer, args.max_target_length)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for index, batch in enumerate(train_loader):
            model.zero_grad()
            x, y = batch
            x = tokenize_source(x)
            input_ids, attention_mask = x.input_ids, x.attention_mask
            input_ids = to_device(input_ids)
            attention_mask = to_device(attention_mask)
            y = to_device(tokenize_target(y)["input_ids"])
            y[y == tokenizer.pad_token_id] = -100

            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=y).loss
            running_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step()
            step = (epoch - 1) * len(train_loader) + index + 1
            if step % 10 == 0:
                avg_loss = running_loss / 10
                report(avg_loss, step)
                checkpoint(epoch, model, args.checkpoints_dir)
                running_loss = 0
        evaluate(args, epoch, model, tokenizer, test_loader)


def checkpoint(epoch_num, model, checkpoint_dir):
    model.save_pretrained(f'{checkpoint_dir}/epoch-{epoch_num}')


def evaluate(args, epoch, model, tokenizer, test_loader):
    model.eval()
    tokenize_source = get_tokenize_fn(tokenizer, args.max_source_length)
    tokenize_target = get_tokenize_fn(tokenizer, args.max_target_length)
    total_loss = 0
    with torch.no_grad():
        for index, batch in enumerate(test_loader):
            x, y = batch
            x = tokenize_source(x)
            input_ids, attention_mask = x.input_ids, x.attention_mask
            input_ids = to_device(input_ids)
            attention_mask = to_device(attention_mask)
            y = to_device(tokenize_target(y)["input_ids"])
            y[y == tokenizer.pad_token_id] = -100

            loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=y).loss
            total_loss += loss.item()
            batch_num = index + 1
            if batch_num % 10 == 0:
                print(
                    f'Epoch: {epoch}, '
                    f'Test batch: up to {batch_num}, '
                    f'Loss: {total_loss / batch_num}')
    model.train()


def report(loss, step):
    print(f'Step: {step}, Loss: {loss}')


def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained('google/flan-t5-base')
    tokenizer.add_tokens([AddedToken(" <="), AddedToken(" <"), AddedToken(" id")])
    model = AutoModelForSeq2SeqLM.from_pretrained('google/flan-t5-base')
    model.resize_token_embeddings(len(tokenizer))
    to_device(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_dataset = Text2SqlDataset(args, "train")
    val_dataset = Text2SqlDataset(args, "val")
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=len(train_loader) * args.epochs)
    print("Training starts...")
    fit(args, model, tokenizer, train_loader, val_loader, optimizer, scheduler)


if __name__ == '__main__':
    main()
