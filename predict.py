from modules.model import *
from transformers.tokenization_bert import BertTokenizer
from processor import NERProcessor, Example
from commons import NERdataset, FeatureExtractor
from torch.utils.data import DataLoader
from underthesea import sent_tokenize, word_tokenize

import torch
import torch.nn as nn
import logging
import argparse


class NER:
    def __init__(self, pretrain_dir="pretrains/baseline/models",
                 feat_dir=None,
                 max_seq_length=256,
                 batch_size=4,
                 device=torch.device('cpu')):

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        processor = NERProcessor(None, self.tokenizer)
        self.fe = FeatureExtractor(dict_dir=feat_dir) if feat_dir is not None else None
        self.label_list = processor.labels
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.device = device
        num_labels = processor.get_num_labels()

        _, self.model, self.feature = model_builder_from_pretrained("bert-base-multilingual-cased",
                                                               num_labels,
                                                               pretrain_dir,
                                                               feat_dir=feat_dir)
        self.model.to(device)

    def convert_sentences_to_features(self, sentences):
        features = []
        for sent_id, sentence in enumerate(sentences):
            if self.fe is None:
                words = " ".join(word_tokenize(sentence))
                ex_words = words.split()
            else:
                ex_words, ex_feats = self.fe.extract_feature(sentence)

            print(f"Input tokens: {ex_words}")
            tokens = []
            feats = {}
            label_ids = []
            token_masks = []

            for i, word in enumerate(ex_words):
                token = self.tokenizer.tokenize(word)
                tokens.extend(token)
                for m in range(len(token)):
                    if m == 0:
                        token_masks.append(1)
                        if self.fe is not None:
                            for feat_key, feat_value in ex_feats[i]:
                                feat_id = self.feature.feature_infos[feat_key]['label'].index(feat_value) + 1
                                if feat_key not in feats:
                                    feats[feat_key] = [feat_id]
                                else:
                                    feats[feat_key].append(feat_id)
                    else:
                        token_masks.append(0)
                        if self.fe is not None:
                            for feat_key, _ in ex_feats[i]:
                                feats[feat_key].append(0)

            if len(tokens) >= self.max_seq_length - 1:
                tokens = tokens[0:(self.max_seq_length - 2)]
                token_masks = token_masks[0:(self.max_seq_length - 2)]
                for k, v in feats.items():
                    feats[k] = v[0:(self.max_seq_length - 2)]

            ntokens = []

            # Add [CLS] token
            ntokens.append("[CLS]")
            token_masks.insert(0, 0)

            if self.fe is not None:
                for feat_key, feat_value in self.feature.special_token["[CLS]"]:
                    feat_id = self.feature.feature_infos[feat_key]['label'].index(feat_value) + 1
                    feats[feat_key].insert(0, feat_id)

            ntokens.extend(tokens)

            # Add [SEP] token
            ntokens.append("[SEP]")
            token_masks.append(0)

            if self.fe is not None:
                for feat_key, feat_value in self.feature.special_token["[CLS]"]:
                    feat_id = self.feature.feature_infos[feat_key]['label'].index(feat_value) + 1
                    feats[feat_key].insert(0, feat_id)

            input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
            attention_masks = [1] * len(input_ids)
            label_masks = [1] * sum(token_masks)
            segment_ids = [0] * self.max_seq_length

            padding = [0] * (self.max_seq_length - len(input_ids))
            input_ids.extend(padding)
            attention_masks.extend(padding)
            token_masks.extend(padding)
            for k in feats.keys():
                feats[k].extend(padding)
            padding = [0] * (self.max_seq_length - len(label_masks))
            label_masks.extend(padding)

            assert len(input_ids) == self.max_seq_length
            assert len(attention_masks) == self.max_seq_length
            assert len(segment_ids) == self.max_seq_length
            assert len(label_masks) == self.max_seq_length
            assert len(token_masks) == self.max_seq_length
            assert sum(token_masks) == sum(label_masks)
            for k in feats.keys():
                assert len(feats[k]) == self.max_seq_length

            features.append(Example(eid=sent_id,
                                    tokens=" ".join(ex_words),
                                    token_ids=input_ids,
                                    attention_masks=attention_masks,
                                    segment_ids=segment_ids,
                                    label_ids=label_ids,
                                    label_masks=label_masks,
                                    token_masks=token_masks,
                                    feats=feats))
        return features

    def preprocess(self, text):
        sentences = sent_tokenize(text)
        features = self.convert_sentences_to_features(sentences)
        data = NERdataset(features, self.device)
        return DataLoader(data, batch_size=self.batch_size)

    def predict(self, text):
        entites = []
        iterator = self.preprocess(text)
        for step, batch in enumerate(iterator):
            sents, token_ids, attention_masks, token_masks, segment_ids, label_ids, label_masks, feats = batch
            logits = self.model(token_ids, attention_masks, token_masks, segment_ids, label_masks, feats)
            logits = torch.argmax(nn.functional.softmax(logits, dim=-1), dim=-1)
            pred = logits.detach().cpu().numpy()
            entity = None
            words = []
            for sent in sents:
                words.extend(sent.split())
            for p, w in list(zip(pred, words)):
                label = self.label_list[p-1]
                if not label == 'O':
                    prefix, label = label.split('-')
                    if entity is None:
                        entity = (w, label)
                    else:
                        if entity[-1] == label:
                            if prefix == 'I':
                                entity = (entity[0] + f' {w}', label)
                            else:
                                entites.append(entity)
                                entity = (w, label)
                        else:
                            entites.append(entity)
                            entity = (w, label)
                elif entity is not None:
                    entites.append(entity)
                    entity = None
                else:
                    entity = None
        return entites


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain_dir", default=None, type=str, required=True)
    parser.add_argument("--feat_dir", default=None, type=str)
    parser.add_argument("--max_seq_length", default=128, type=int)
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--cuda", action="store_true")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    ner = NER(args.pretrain_dir, args.feat_dir, args.max_seq_length, args.batch_size, device)

    while True:
        input_text = input("Enter text: ")
        if len(input_text.strip()) == 0:
            print("Input NULL, auto use text sample!!!!")
            input_text = """Ông Nguyễn Đức Vinh - giám đốc Sở Nông nghiệp và phát triển nông thôn Hà Giang - nhận 
            định mưa lớn cục bộ trong thời gian ngắn là nguyên nhân chính dẫn đến tình trạng ngập lụt chưa từng có ở 
            TP Hà Giang. Mặt khác, theo ông Vinh, TP Hà Giang có địa hình lòng chảo, bao xung quanh là núi cũng khiến 
            lượng nước đổ dồn về trung tâm rất lớn."""
        print(ner.predict(input_text))

