import os
import csv
import sys

maxInt = sys.maxsize

while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)


class Example:
    def __init__(self, eid: int, token_ids: list, token_mask: list, segment_ids: list,
                 label_ids: list, label_mask: list):
        self.eid = eid
        self.token_ids = token_ids
        self.token_mask = token_mask
        self.segment_ids = segment_ids
        self.label_ids = label_ids
        self.label_mask = label_mask


class NERProcessor:
    def __init__(self, data_dir: str, tokenizer):
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.labels = ["O", "B-MISC", "I-MISC",  "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]

    def get_labels(self):
        return self.labels

    def get_example(self, data_type: str = "train"):
        if data_type == "train":
            return self._read_file(os.path.join(self.data_dir, 'train.csv'))
        elif data_type == "dev":
            return self._read_file(os.path.join(self.data_dir, 'dev.csv'))
        elif data_type == "test":
            return self._read_file(os.path.join(self.data_dir, 'test.csv'))
        else:
            print(f"ERROR: {data_type} not found!!!")

    @staticmethod
    def _read_file(file_path: str):
        """Reads a tab separated value file."""
        with open(file_path, "r", encoding="utf-8") as f:
            reader = csv.reader(f, delimiter="\t")
            eid = 0
            words = []
            labels = []
            examples = []
            for line in reader:
                if len(line) >= 2:
                    words.append(line[0].strip())
                    labels.append(line[-1].strip())
                else:
                    examples.append((eid, words, labels))
                    words = []
                    labels = []
                    eid += 1
            return examples

    def convert_examples_to_features(self, examples, max_seq_length):
        features = []
        for (ex_index, example) in enumerate(examples):
            tokens = []
            labels = []
            label_mask = []
            for i, (word, label) in enumerate(zip(example[1], example[-1])):
                token = self.tokenizer.tokenize(word)
                tokens.extend(token)
                label_1 = label
                for m in range(len(token)):
                    if m == 0:
                        labels.append(label_1)
                        label_mask.append(1)
                    else:
                        label_mask.append(0)
            if len(tokens) >= max_seq_length - 1:
                tokens = tokens[0:(max_seq_length - 2)]
                labels = labels[0:(max_seq_length - 2)]
                label_mask = label_mask[0:(max_seq_length - 2)]
            ntokens = []
            segment_ids = []
            ntokens.append("[CLS]")
            label_ids = []
            segment_ids.append(0)
            label_mask.insert(0, 0)
            label_ids.append(self.labels.index("O")+1)
            for i, token in enumerate(tokens):
                ntokens.append(token)
                segment_ids.append(0)
                if len(labels) > i:
                    label_ids.append(self.labels.index(labels[i])+1)
            ntokens.append("[SEP]")
            segment_ids.append(0)
            label_mask.append(0)
            label_ids.append(self.labels.index("O")+1)
            input_ids = self.tokenizer.convert_tokens_to_ids(ntokens)
            input_mask = [1] * len(input_ids)
            while len(input_ids) < max_seq_length:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                label_ids.append(0)
                label_mask.append(0)
            while len(label_ids) < max_seq_length:
                label_ids.append(0)
            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length
            assert len(label_ids) == max_seq_length
            assert len(label_mask) == max_seq_length

            if ex_index < 5:
                print("*** Example ***")
                print("guid: %s" % (example[0]))
                print("tokens: %s" % " ".join([str(x) for x in tokens]))
                print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
                print(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                print("label: %s" % " ".join([str(x) for x in label_ids]))
                print("label_mask: %s" % " ".join([str(x) for x in label_mask]))
            features.append(
                Example(eid=example[0],
                        token_ids=input_ids,
                        token_mask=input_mask,
                        segment_ids=segment_ids,
                        label_ids=label_ids,
                        label_mask=label_mask))

        return features


if __name__ == "__main__":
    from transformers.tokenization_bert import BertTokenizer
    tokenzier = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    processor = NERProcessor("./dataset", tokenzier)
    a = processor.get_example("train")
    features = processor.convert_examples_to_features(a, 126)
    print()