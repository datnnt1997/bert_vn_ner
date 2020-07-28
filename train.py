import os
import argparse
import random
import numpy as np

from torch.utils.data import DataLoader, RandomSampler
from torch.utils.tensorboard import SummaryWriter
from modules.model import *
from commons import NERdataset, logger, init_logger
from processor import NERProcessor
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers.tokenization_bert import BertTokenizer
from sklearn.metrics import classification_report, f1_score


def build_dataset(args, processor, data_type='train', feature=None, device=torch.device('cpu')):
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
        examples = processor.get_example(data_type, feature is not None)

        features = processor.convert_examples_to_features(examples, args.max_seq_length, feature)
        print("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)
    return NERdataset(features, device)


def caculator_metric(preds, golds, labels):
    pred_iob_labels = [labels[label_id - 1] for label_id in preds]
    gold_iob_labels = [labels[label_id - 1] for label_id in golds]

    pred_labels = [labels[label_id - 1].split("-")[-1].strip() for label_id in preds]
    gold_labels = [labels[label_id - 1].split("-")[-1].strip() for label_id in golds]

    iob_metric = classification_report(pred_iob_labels, gold_iob_labels, output_dict=True)
    metric = classification_report(pred_labels, gold_labels, output_dict=True)

    return iob_metric, metric


def update_model_weights(model, iterator, optimizer, scheduler):
    # init static variables
    tr_loss = 0
    model.train()

    for step, batch in enumerate(tqdm(iterator, desc="Iteration")):
        tokens, token_ids, attention_masks, token_mask, segment_ids, label_ids, label_masks, feats = batch
        loss, _ = model.calculate_loss(token_ids, attention_masks, token_mask, segment_ids, label_ids, label_masks,
                                       feats)
        tr_loss += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
        if (step + 1) % args.gradient_accumulation_steps == 0:
            optimizer.step()
            scheduler.step()
            model.zero_grad()
    return tr_loss


def evaluate(model, iterator, label_map):
    # init static variables
    preds = []
    golds = []
    eval_loss = 0
    model.eval()

    for step, batch in enumerate(tqdm(iterator, desc="Iteration")):
        tokens, token_ids, attention_masks, token_mask, segment_ids, label_ids, label_masks, feats = batch
        loss, (logits, labels) = model.calculate_loss(token_ids, attention_masks, token_mask, segment_ids, label_ids,
                                                      label_masks, feats)
        eval_loss += loss.item()
        logits = torch.argmax(nn.functional.softmax(logits, dim=-1), dim=-1)
        pred = logits.detach().cpu().numpy()
        gold = labels.to('cpu').numpy()
        preds.extend(pred)
        golds.extend(gold)

    iob_metric, metric = caculator_metric(preds, golds, label_map)

    return eval_loss, iob_metric, metric


def run(args):
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    summary_writer = SummaryWriter(args.log_dir)
    init_logger(f"{args.output_dir}/vner_trainning.log")

    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    processor = NERProcessor(args.data_dir, tokenizer)
    num_labels = processor.get_num_labels()
    logger.info("Build model ...")
    config, model, feature = model_builder(model_name_or_path=args.model_name_or_path,
                                           num_labels=num_labels,
                                           feat_config_path=args.feat_config,
                                           one_hot_embed=args.one_hot_emb,
                                           use_lstm=args.use_lstm,
                                           device=device)
    model.to(device)
    logger.info("Prepare dataset ...")
    train_data = build_dataset(args, processor, data_type='train', feature=feature, device=device)
    eval_data = build_dataset(args, processor, data_type='test', feature=feature, device=device)

    train_sampler = RandomSampler(train_data)
    train_iterator = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size)

    eval_sampler = RandomSampler(eval_data)
    eval_iterator = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    num_train_optimization_steps = len(train_iterator) // args.gradient_accumulation_steps * args.num_train_epochs
    warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=num_train_optimization_steps)

    logger.info("="*30 + f"Summary" + "="*30)
    logger.info("MODEL:")
    logger.info(f"\tBERT model: {args.model_name_or_path}")
    logger.info(f"\tNumber of parameters: {sum(p.numel() for p in model.parameters())}")
    logger.info("DATASET:")
    logger.info(f"\tNumber of train Examples: {len(train_data)}")
    logger.info(f"\tNumber of eval Examples: {len(eval_data)}")
    logger.info(f"\tNumber of labels: {len(processor.labels)}")
    logger.info("Hyper-Parameters:")
    logger.info(f"\tMax sequence length: {args.max_seq_length}")
    logger.info(f"\tLearning rate: {args.learning_rate}")
    logger.info(f"\tNumber of epochs: {args.num_train_epochs}")
    logger.info(f"\tTrain batch size: {args.train_batch_size}")
    logger.info(f"\tEval batch size: {args.eval_batch_size}")
    logger.info(f"\tAdam epsilon: {args.adam_epsilon}")
    logger.info(f"\tWeight decay: {args.weight_decay}")
    logger.info(f"\tWarmup Proportion: {args.warmup_proportion}")
    logger.info(f"\tMax grad norm: {args.max_grad_norm}")
    logger.info(f"\tGradient accumulation steps: {args.gradient_accumulation_steps}")
    logger.info(f"\tSeed: {args.seed}")
    logger.info(f"\tCuda: {args.cuda}")
    logger.info(f"\tFeat config: {args.feat_config}")
    logger.info(f"\tUse one-hot embbeding: {args.one_hot_emb}")
    logger.info(f"\tOutput directory: {args.output_dir}")
    logger.info(f"\tLog directory: {args.log_dir}")


    model.train()
    best_score = -1
    for e in range(int(args.num_train_epochs)):
        logger.info("="*30 + f"Epoch {e}" + "="*30)
        tr_loss = update_model_weights(model, train_iterator, optimizer, scheduler)
        logger.info(f"train Loss: {tr_loss}")
        eval_loss, iob_metric, metric = evaluate(model, eval_iterator, processor.labels)
        logger.info(f"eval Loss: {eval_loss}")
        logger.info(f"F1-Score tag: {metric['macro avg']['f1-score']}")
        logger.info(f"F1-Score IOB-tag: {iob_metric['macro avg']['f1-score']}")
        logger.info(f"Metric:")
        logger.info(f"\tO: {metric['O']['f1-score'] if 'O' in metric else 0.0}")
        logger.info(f"\tMISC: {metric['MISC']['f1-score'] if 'MISC' in metric else 0.0}")
        logger.info(f"\tPER: {metric['PER']['f1-score'] if 'PER' in metric else 0.0}")
        logger.info(f"\tORG: {metric['ORG']['f1-score'] if 'ORG' in metric else 0.0}")
        logger.info(f"\tLOC: {metric['LOC']['f1-score'] if 'LOC' in metric else 0.0}")
        summary_writer.add_scalar('LOSS/TRAIN', tr_loss, e)
        summary_writer.add_scalar('LOSS/EVAL', eval_loss, e)
        summary_writer.add_scalars('F1-SCORE/TAG', {
            "AVG": metric['macro avg']['f1-score'],
            "O": metric['O']['f1-score'] if 'O' in metric else 0.0,
            "MISC": metric['MISC']['f1-score'] if 'MISC' in metric else 0.0,
            "PER": metric['PER']['f1-score'] if 'PER' in metric else 0.0,
            "ORG": metric['ORG']['f1-score'] if 'ORG' in metric else 0.0,
            "LOC": metric['LOC']['f1-score'] if 'LOC' in metric else 0.0
        }, e)
        summary_writer.add_scalars('F1-SCORE/IOB TAG', {
            "AVG": iob_metric['macro avg']['f1-score'],
            "O": iob_metric['O']['f1-score'] if 'O' in iob_metric else 0.0,
            "B-MISC": iob_metric['B-MISC']['f1-score'] if 'B-MISC' in iob_metric else 0.0,
            "I-MISC": iob_metric['I-MISC']['f1-score'] if 'I-MISC' in iob_metric else 0.0,
            "B-PER": iob_metric['B-PER']['f1-score'] if 'B-PER' in iob_metric else 0.0,
            "I-PER": iob_metric['I-PER']['f1-score'] if 'I-PER' in iob_metric else 0.0,
            "B-ORG": iob_metric['B-ORG']['f1-score'] if 'B-ORG' in iob_metric else 0.0,
            "I-ORG": iob_metric['I-ORG']['f1-score'] if 'I-ORG' in iob_metric else 0.0,
            "B-LOC": iob_metric['B-LOC']['f1-score'] if 'B-LOC' in iob_metric else 0.0,
            "I-LOC": iob_metric['I-LOC']['f1-score'] if 'I-LOC' in iob_metric else 0.0
        }, e)

        if iob_metric['macro avg']['f1-score'] > best_score:
            best_score = iob_metric['macro avg']['f1-score']
            best_epoch = e
            model_path = f"{args.output_dir}/vner_model.bin"
            torch.save(model.state_dict(), model_path)
            logger.info(f"Model save at epoch {best_epoch} with best score {best_score}")
            summary_writer.add_text("Best result", f"F1-Score: {best_score}", best_epoch)
            summary_writer.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True)
    parser.add_argument("--model_name_or_path", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--log_dir", default=None, type=str, required=True)

    # Other parameters
    parser.add_argument("--feat_config", default=None, type=str)
    parser.add_argument("--one_hot_emb", action='store_true')
    parser.add_argument("--use_lstm", action='store_true')
    parser.add_argument("--cache_dir", default="", type=str)
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--train_batch_size", default=8, type=int)
    parser.add_argument("--eval_batch_size", default=4, type=int)
    parser.add_argument("--learning_rate", default=2e-5, type=float)
    parser.add_argument("--weight_decay", default=0.0, type=float)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float)
    parser.add_argument("--num_train_epochs", default=100.0, type=float)
    parser.add_argument("--warmup_proportion", default=0.1, type=float)
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--cuda", action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    run(args)

