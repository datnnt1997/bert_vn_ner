import os
import torch
import argparse
import random
import numpy as np
import json
import torch.functional as F
from model import *
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from processor import NERProcessor
from tqdm import tqdm
from transformers import AdamW
from sklearn.metrics import classification_report, f1_score


def build_dataset(args, processor, data_type='train'):
    # Load data features from cache or dataset file
    cached_features_file = os.path.join(args.data_dir, 'cached_{}_{}_{}'.format(
        data_type,
        list(filter(None, args.model_name_or_path.split('/'))).pop(),
        str(args.max_seq_length)))
    if os.path.exists(cached_features_file):
        print("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        print("Creating features from dataset file at %s", args.data_dir)
        examples = processor.get_example(data_type)

        features = processor.convert_examples_to_features(examples, args.max_seq_length)
        print("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    # Convert to Tensors and build dataset
    all_input_ids = torch.tensor([f.token_ids for f in features], dtype=torch.long)
    all_token_mask = torch.tensor([f.token_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_label_id = torch.tensor([f.label_ids for f in features], dtype=torch.long)
    all_label_mask = torch.tensor([f.label_mask for f in features], dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_token_mask, all_segment_ids, all_label_id, all_label_mask)
    return dataset


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True,
                        help="Path to pre-trained model or shortcut name selected in the list")

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=4, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=100.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cuda else "cpu")
    n_gpu = torch.cuda.device_count()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    processor = NERProcessor(args.data_dir, tokenizer)
    label_list = processor.get_labels()
    num_labels = len(label_list) + 1

    config, model = modelbuilder(args.model_name_or_path, num_labels)

    train_data = build_dataset(args, processor, data_type='train')

    model.to(device)
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    # warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    # Train
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    model.train()
    best_score = -1
    best_epoch = 0
    training_loss = []
    evaling_loss = []
    for e in range(int(args.num_train_epochs)):
        print("="*30 + f"Epoch {e}" + "="*30)
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        model.train()
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, l_mask = batch
            loss, _ = model.calculate_loss(input_ids, segment_ids, input_mask, label_ids, l_mask)

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            loss.backward()
            optimizer.step()
            model.zero_grad()
        print(tr_loss)
        training_loss.append(tr_loss)
        eval_data = build_dataset(args, processor, data_type='test')
        eval_sampler = RandomSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
        model.eval()
        eval_loss = 0
        nb_eval_examples, nb_eval_steps = 0, 0
        preds = []
        golds = []
        for step, batch in enumerate(tqdm(eval_dataloader, desc="Iteration")):
            batch = tuple(t.to(device) for t in batch)
            input_ids, input_mask, segment_ids, label_ids, l_mask = batch
            loss, logits = model.calculate_loss(input_ids, segment_ids, input_mask, label_ids, l_mask)

            eval_loss += loss.item()
            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

            logits = torch.argmax(nn.functional.softmax(logits, dim=-1), dim=-1)
            pred = logits.detach().cpu().numpy()
            l_mask = l_mask.view(-1) == 1
            label_ids = label_ids.view(-1)[l_mask]
            gold = label_ids.to('cpu').numpy()
            preds.extend(pred)
            golds.extend(gold)
        metric = classification_report(golds, preds)
        f1 = f1_score(golds, preds, average="macro")
        evaling_loss.append(eval_loss)
        print(metric)
        print(eval_loss)
        if f1 > best_score:
            best_score = f1
            best_epoch = e
            model_path = f"{args.output_dir}/vner_model.bin"
            torch.save(model.state_dict(), model_path)
            print(f"Model save at epoch {best_epoch} with best score {best_score}")
        history_path = f"{args.output_dir}/history.json"
        history = {"train_loss": training_loss,
                   "eval_loss": evaling_loss,
                   "best_epoch": best_epoch,
                   "best_f1": best_score}
        json.dump(history, history_path)

if __name__ == "__main__":
    main()

