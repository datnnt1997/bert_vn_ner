from modules.model import *
from transformers.tokenization_bert import BertTokenizer
from processor import NERProcessor
from vncorenlp import VnCoreNLP

import torch
import torch.nn as nn


class NER:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
        processor = NERProcessor(None, self.tokenizer)
        self.label_list = processor.get_labels()
        self.max_seq_length = 256
        num_labels = len(self.label_list) + 1

        _, self.model = model_builder_from_pretrained("bert-base-multilingual-cased", num_labels, "outputs")
        self.vncore = VnCoreNLP("VnCoreNLP-master/VnCoreNLP-1.1.1.jar", annotators="wseg",
                                      max_heap_size='-Xmx500m')

    def preprocess(self, text):
        words = self.vncore.tokenize(text)[0]
        print(words)
        tokens = []
        label_mask = []
        for word in words:
            token = self.tokenizer.tokenize(word)
            tokens.extend(token)
            for m in range(len(token)):
                if m == 0:
                    label_mask.append(1)
                else:
                    label_mask.append(0)
        if len(tokens) >= self.max_seq_length - 1:
            tokens = tokens[0:(self.max_seq_length - 2)]
            label_mask = label_mask[0:(self.max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        label_mask.insert(0, 0)
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
        ntokens.append("[SEP]")
        segment_ids.append(0)
        label_mask.append(0)
        input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        while len(input_ids) < self.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            label_mask.append(0)
        assert len(input_ids) == self.max_seq_length
        assert len(input_mask) == self.max_seq_length
        assert len(segment_ids) == self.max_seq_length
        assert len(label_mask) == self.max_seq_length

        input_id_tensors = torch.tensor([input_ids], dtype=torch.long)
        all_token_mask_tensor = torch.tensor([input_mask], dtype=torch.long)
        all_segment_id_tensors = torch.tensor([segment_ids], dtype=torch.long)
        all_label_mask_tensors = torch.tensor([label_mask], dtype=torch.long)
        return (words, input_id_tensors, all_token_mask_tensor, all_segment_id_tensors, all_label_mask_tensors)

    def predict(self, text):
        words, input_ids, input_mask, segment_ids, l_mask = self.preprocess(text)
        logits = self.model(input_ids, segment_ids, input_mask, l_mask)
        logits = torch.argmax(nn.functional.softmax(logits, dim=-1), dim=-1)
        pred = logits.detach().cpu().numpy()
        labels = []
        for p in pred:
            labels.append(self.label_list[p-1])
        print(labels)
        return words, labels


if __name__ == "__main__":
    ner = NER()
    while True:
        input_text = input(">>>>")
        ner.predict(input_text)

