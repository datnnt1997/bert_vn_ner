from commons import FeatureExtractor

import os
import csv
import argparse
import sys

maxInt = sys.maxsize
while True:
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)


def read_csv(file_path: str) -> list:
    with open(file_path, 'r', encoding="utf-8") as file:
        lines = file.readlines()
        words = []
        pos_tags = []
        labels = []
        examples = []
        for line in lines:
            items = line.split("\t")
            if len(items) >= 2:
                words.append(items[0].strip())
                pos_tags.append(items[1].strip())
                labels.append(items[-1].strip())
            else:
                examples.append((words, pos_tags, labels))
                words = []
                pos_tags = []
                labels = []
        return examples


def create_example_with_features(feature_extractor: FeatureExtractor, examples: list):
    fe_examples = []
    for example in examples:
        words, pos_tags, labels = example
        fe_example = feature_extractor.extract_feature(words, pos_tags, labels)
        fe_examples.append(fe_example)
    return fe_examples


def write_example(output_file: str, examples: list):
    with open(output_file, 'w', encoding='utf-8') as writer:
        for example in examples:
            for sample in example:
                writer.write(f"{sample}\n")
            writer.write("\n")


def preprocess():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--dict_dir", default=None, type=str, required=True)
    parser.add_argument("--feature_types", default="pos,cf,sc,fw,qb,num,loc,org,per,ppos", type=str)
    args = parser.parse_args()
    fe_types = args.feature_types.split(",")

    fe = FeatureExtractor(dict_dir=args.dict_dir,
                          feature_types=fe_types)

    # Process Train file
    print("Train dataset Processing ...")
    train_examples = create_example_with_features(fe, read_csv(os.path.join(args.data_dir, 'train.csv')))
    write_example(os.path.join(args.output_dir, 'train.csv'), train_examples)

    # Process Dev file
    print("Dev dataset Processing ...")
    dev_examples = create_example_with_features(fe, read_csv(os.path.join(args.data_dir, 'dev.csv')))
    write_example(os.path.join(args.output_dir, 'dev.csv'), dev_examples)

    # Process Test file
    print("Test dataset Processing ...")
    test_examples = create_example_with_features(fe, read_csv(os.path.join(args.data_dir, 'test.csv')))
    write_example(os.path.join(args.output_dir, 'test.csv'), test_examples)


if __name__ == "__main__":
    preprocess()
