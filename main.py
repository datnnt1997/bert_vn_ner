import os
import torch
import argparse
import random
import numpy as np
import torch.functional as F
from model import *
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from processor import NERProcessor
from tqdm import tqdm
from transformers import AdamW

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
        if args.local_rank in [-1, 0]:
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

    parser.add_argument("--vocab_file", default='', type=str)

    parser.add_argument("--spm_model_file", default='', type=str)

    ## Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_predict", action='store_true',
                        help="Whether to run the model in inference mode on the test set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--train_batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=2, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=1.0, type=float,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for,E.g., 0.1 = 10% of training.")
    parser.add_argument('--logging_steps', type=int, default=10,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=1000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")

    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
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

    # num_train_optimization_steps = int(
    #    len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs

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

    # # Train
    # train_sampler = RandomSampler(train_data)
    # train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)
    # model.train()
    # for _ in range(int(args.num_train_epochs)):
    #     tr_loss = 0
    #     nb_tr_examples, nb_tr_steps = 0, 0
    #     for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
    #         batch = tuple(t.to(device) for t in batch)
    #         input_ids, input_mask, segment_ids, label_ids, l_mask = batch
    #         loss, _ = model.calculate_loss(input_ids, segment_ids, input_mask, label_ids, l_mask)
    #
    #         loss = loss / args.gradient_accumulation_steps
    #         loss.backward()
    #
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
    #
    #         tr_loss += loss.item()
    #         nb_tr_examples += input_ids.size(0)
    #         nb_tr_steps += 1
    #         if (step + 1) % args.gradient_accumulation_steps == 0:
    #             optimizer.step()
    #             model.zero_grad()
    #     print(tr_loss)
    eval_data = build_dataset(args, processor, data_type='test')
    eval_sampler = RandomSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    model.eval()
    eval_loss = 0
    nb_eval_examples, nb_eval_steps = 0, 0
    for step, batch in enumerate(tqdm(eval_dataloader, desc="Iteration")):
        batch = tuple(t.to(device) for t in batch)
        input_ids, input_mask, segment_ids, label_ids, l_mask = batch
        loss, logits = model.calculate_loss(input_ids, segment_ids, input_mask, label_ids, l_mask)

        loss = loss / args.gradient_accumulation_steps
        eval_loss += loss.item()
        nb_eval_examples += input_ids.size(0)
        nb_eval_steps += 1
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

        logits = torch.argmax(F.softmax(logits, dim=2), dim=2)
        logits = logits.detach().cpu().numpy()
        label_ids = label_ids.to('cpu').numpy()
        input_mask = input_mask.to('cpu').numpy()

    print(tr_loss)

if __name__ == "__main__":
    main()

