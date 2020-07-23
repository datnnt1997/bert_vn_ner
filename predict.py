from modules.model import *
from transformers.tokenization_bert import BertTokenizer
from processor import NERProcessor, Example
from commons import NERdataset
from underthesea import sent_tokenize, word_tokenize
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import logging


class NER:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        processor = NERProcessor(None, self.tokenizer)
        self.label_list = processor.labels
        self.max_seq_length = 256
        self.batch_size = 4
        self.device = torch.device('cpu')
        num_labels = processor.get_num_labels()

        _, self.model, feature = model_builder_from_pretrained("bert-base-multilingual-cased", num_labels, "outputs")

    def convert_sentences_to_features(self, sentences):
        features = []
        for sent_id, sentence in enumerate(sentences):
            words = sentence.split()
            logging.info(f"Input tokens: {words}")
            tokens = []
            feats = {}
            label_ids = []
            token_masks = []
            for word in words:
                token = self.tokenizer.tokenize(word)
                tokens.extend(token)
                for m in range(len(token)):
                    if m == 0:
                        token_masks.append(1)
                    else:
                        token_masks.append(0)

            if len(tokens) >= self.max_seq_length - 1:
                tokens = tokens[0:(self.max_seq_length - 2)]
                token_masks = token_masks[0:(self.max_seq_length - 2)]

            ntokens = []
            segment_ids = []
            # Add [CLS] token
            ntokens.append("[CLS]")
            token_masks.insert(0, 0)
            ntokens.extend(tokens)
            # Add [SEP] token
            ntokens.append("[SEP]")
            segment_ids.append(0)
            token_masks.append(0)
            input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)

            attention_masks = [1] * len(input_ids)
            label_masks = [1] * sum(token_masks)
            segment_ids = [0] * self.max_seq_length

            padding = [0] * (self.max_seq_length - len(input_ids))
            input_ids.extend(padding)
            attention_masks.extend(padding)
            token_masks.extend(padding)

            padding = [0] * (self.max_seq_length - len(label_masks))
            label_masks.extend(padding)
            assert len(input_ids) == self.max_seq_length
            assert len(attention_masks) == self.max_seq_length
            assert len(segment_ids) == self.max_seq_length
            assert len(label_masks) == self.max_seq_length
            assert len(token_masks) == self.max_seq_length
            assert sum(token_masks) == sum(label_masks)

            features.append(Example(eid=sent_id,
                                    tokens=sentence,
                                    token_ids=input_ids,
                                    attention_masks=attention_masks,
                                    segment_ids=segment_ids,
                                    label_ids=label_ids,
                                    label_masks=label_masks,
                                    token_masks=token_masks,
                                    feats=feats))
        return features

    def preprocess(self, text, feats=None):
        sentences = sent_tokenize(' '.join(word_tokenize(text)))
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
            tokens = []
            [tokens.extend(sent.split()) for sent in sents]
            for p, w in list(zip(pred, tokens)):
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
        print(entites)
        return entites


if __name__ == "__main__":
    ner = NER()
    while True:
        input_text = input("Enter text: ")
        # input_text = """Theo quy định của Trung Quốc, khi lưu lượng nước đổ về hồ chứa Tam Hiệp đạt 50.000m3/s và mực nước ở trạm Liên Hoa Đường tăng lên mức cảnh báo, lũ sẽ bắt đầu được đánh số. Trước đó, trận lũ số 1 đã hình thành ở thượng nguồn sông Trường Giang vào đầu tháng 7."""
        ner.predict(input_text)

