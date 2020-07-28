from logging.handlers import RotatingFileHandler
from torch.utils.data.dataset import Dataset
from underthesea import pos_tag, word_tokenize

import os
import re
import string
import json
import logging
import torch

logger = logging.getLogger()


def init_logger(log_file=None, log_file_level=logging.NOTSET):
    log_format = logging.Formatter("%(message)s")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.handlers = [console_handler]

    if log_file and log_file != '':
        file_handler = RotatingFileHandler(
            log_file)
        file_handler.setLevel(log_file_level)
        file_handler.setFormatter(log_format)
        logger.addHandler(file_handler)
    return logger


class NERdataset(Dataset):
    def __init__(self, features, device):
        self.features = features
        self.device = device

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        sample = self.features[idx]
        token_id_tensors = torch.tensor(sample.token_ids, dtype=torch.long).to(device=self.device)
        attention_mask_tensors = torch.tensor(sample.attention_masks, dtype=torch.long).to(device=self.device)
        token_mask_tensors = torch.tensor(sample.token_masks, dtype=torch.long).to(device=self.device)
        segment_id_tensors = torch.tensor(sample.segment_ids, dtype=torch.long).to(device=self.device)
        label_id_tensors = torch.tensor(sample.label_ids, dtype=torch.long).to(device=self.device)
        label_mask_tensors = torch.tensor(sample.label_masks, dtype=torch.long).to(device=self.device)

        feat_tensors = {}
        for feat_key, feat_value in sample.feats.items():
            feat_tensors[feat_key] = torch.tensor(feat_value, dtype=torch.long).to(device=self.device)
        return sample.tokens, token_id_tensors, attention_mask_tensors, token_mask_tensors, segment_id_tensors, \
               label_id_tensors, label_mask_tensors, feat_tensors


def pos_tag_normalize(tag):
    tags_map = {
        "Ab": "A",
        "B": "FW",
        "Cc": "C",
        "Fw": "FW",
        "Nb": "FW",
        "Ne": "Nc",
        "Ni": "Np",
        "NNP": "Np",
        "Ns": "Nc",
        "S": "Z",
        "Vb": "V",
        "Y": "Np"
    }
    if tag in tags_map:
        return tags_map[tag]
    else:
        return tag


class Feature:
    def __init__(self, config_file: str, one_hot_emb: bool = True):
        self.feature_infos = None
        self.special_token = None
        self.num_of_feature = 0
        self.feature_keys = []
        self.one_hot_emb = one_hot_emb
        self.feature_emb_dim = 0
        self.read_config(config_file)

    def read_config(self, config_file: str):
        with open(config_file, "r", encoding="utf-8") as f:
            config = json.load(f)
            self.feature_infos = {}
            for key in config["feats"].keys():
                self.feature_infos[key] = config["feats"][key]
                self.feature_infos[key]['size'] = len(config["feats"][key]['label'])
                self.feature_emb_dim += config["feats"][key]['dim']
            self.special_token = config["special_token"]
            self.num_of_feature = len(self.feature_infos)
            self.feature_keys = list(self.feature_infos.keys())
            f.close()


class FeatureExtractor:
    def __init__(self, dict_dir: str, feature_types: list = None):
        if feature_types is None:
            feature_types = ["pos", "cf", "sc", "fw", "qb", "num", "loc", "org", "per", "ppos"]
        self.feature_types = feature_types
        self.loc_dict = None
        self.org_dict = None
        self.per_dict = None
        self.ppos_dict = None

        self.loc_max_lenght = 0
        self.org_max_lenght = 0
        self.per_max_lenght = 0
        self.ppos_max_lenght = 0

        self.load_dict(dict_dir)

    @staticmethod
    def read_dict(dict_path: str) -> (list, int):
        feature_dict = set()
        feature_max_lenght = 0
        with open(dict_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line_text = line.strip().lower()
                line_text_length = len(line_text.split())
                feature_dict.add(line_text)
                if line_text_length > feature_max_lenght:
                    feature_max_lenght = line_text_length
            f.close()
        return feature_dict, feature_max_lenght

    def load_dict(self, dict_dir: str):
        # Load Location dictionary
        self.loc_dict, self.loc_max_lenght = self.read_dict(os.path.join(dict_dir, 'vnLocation.txt'))
        # Load Location dictionary
        self.org_dict, self.org_max_lenght = self.read_dict(os.path.join(dict_dir, 'vnOrganization.txt'))
        # Load Location dictionary
        self.per_dict, self.per_max_lenght = self.read_dict(os.path.join(dict_dir, 'vnFullNames.txt'))
        # Load Location dictionary
        self.ppos_dict, self.ppos_max_lenght = self.read_dict(os.path.join(dict_dir, 'vnPersonalPositions.txt'))

    @staticmethod
    def wseg_and_add_pos_tag_feature(sentence: str or list,
                                     pos_tags: list = None,
                                     ner_labels: list = None) -> (list, list):
        if type(sentence) == str:
            sentence = sentence.split()
        if ner_labels is None:
            ner_labels = ['O'] * len(sentence)
        pos_features = []
        words = []
        labels = []

        if pos_tags is None:
            annotated_text = pos_tag(" ".join(sentence))
            sentence = []
            pos_tags = []
            for word, pos in annotated_text:
                sentence.append(word.strip())
                pos_tags.append(pos.strip())
            ner_labels = ['O'] * len(sentence)

        for word, pos, label in list(zip(sentence, pos_tags, ner_labels)):
            tokens = word.split()
            (prefix, tag) = label.split('-') if not label == 'O' else ('', label)
            for idx, token in enumerate(tokens):
                if token.strip() == '':
                    continue
                if idx == 0 and prefix.strip() == 'B':
                    labels.append(label)
                else:
                    labels.append(f'I-{tag.strip()}' if not label == 'O' else 'O')
                words.append(token.strip())
                pos_features.append('[POS]' + pos_tag_normalize(pos.strip()))
        return words, pos_features, labels

    @staticmethod
    def word_segment(sentence: str or list, ner_labels: list = None):
        if type(sentence) == str:
            sentence = word_tokenize(sentence)
        if ner_labels is None:
            ner_labels = ['O'] * len(sentence)
        words = []
        labels = []
        for word, label in list(zip(sentence, ner_labels)):
            tokens = word.split()
            prefix, tag = label.split('-') if not label == 'O' else '', label
            for idx, token in enumerate(tokens):
                if token.strip() == '':
                    continue
                if idx == 0 and prefix.strip() == 'B':
                    labels.append(label)
                else:
                    labels.append(f'I-{tag.strip()}' if not label == 'O' else 'O')
                words.append(token)
        return words, labels

    @staticmethod
    def add_case_feature(sentence: list) -> list:
        case_features = []
        for word in sentence:
            if word.isupper():
                case_features.append('[Case]A_Cap')
            elif word[0].isupper():
                case_features.append('[Case]I_Cap')
            elif any(c.isupper() for c in word):
                case_features.append('[Case]M_Cap')
            else:
                case_features.append('[Case]N_Cap')
        return case_features

    @staticmethod
    def add_sequence_case_feature(sentence: list) -> list:
        sc_features = []
        for idx, word in enumerate(sentence):
            if len(sentence) == 1:
                sc_features.append('[SC]N_Cap')
                continue
            if idx == 0:
                if sentence[idx + 1][0].isupper():
                    sc_features.append('[SC]Pos_Cap')
                else:
                    sc_features.append('[SC]N_Cap')
            elif idx == (len(sentence) - 1):
                if sentence[idx - 1][0].isupper():
                    sc_features.append('[SC]Pre_Cap')
                else:
                    sc_features.append('[SC]N_Cap')
            else:
                if sentence[idx + 1][0].isupper() and sentence[idx - 1][0].isupper():
                    if sentence[idx][0].isupper():
                        sc_features.append('[SC]I_Cap')
                    else:
                        sc_features.append('[SC]NI_Cap')
                elif sentence[idx + 1][0].isupper():
                    sc_features.append('[SC]Pos_Cap')
                elif sentence[idx - 1][0].isupper():
                    sc_features.append('[SC]Pre_Cap')
                else:
                    sc_features.append('[SC]N_Cap')
        return sc_features

    @staticmethod
    def add_fisrt_word_feature(sentence: list) -> list:
        fw_features = []
        for idx in range(len(sentence)):
            if idx == 0:
                fw_features.append('[FW]1')
            elif sentence[idx - 1] in '.!?...':
                fw_features.append('[FW]1')
            else:
                fw_features.append('[FW]0')
        return fw_features

    @staticmethod
    def add_quotes_brackets_feature(sentence: list) -> list:
        qb_features = []
        isquotes = False
        isbrackets = False
        for word in sentence:
            if word == '"' or word == '“' or word == '”':
                isquotes = not isquotes
                qb_features.append('[QB]0')
                continue
            elif word == '(':
                isbrackets = True
                qb_features.append('[QB]0')
                continue
            elif word == ')':
                isbrackets = False
                qb_features.append('[QB]0')
                continue
            if isquotes is True or isbrackets is True:
                qb_features.append('[QB]1')
            else:
                qb_features.append('[QB]0')
        return qb_features

    @staticmethod
    def add_number_feature(sentence: list) -> list:
        num_features = []
        num = ['một', 'hai', 'ba', 'bốn', 'năm', 'sáu', 'bảy', 'tám', 'chín', 'mười', 'chục', 'trăm', 'nghìn', 'vạn',
               'triệu', 'tỷ']
        for word in sentence:
            if word.lower() in num or re.sub(f'[{string.punctuation}]', '', word).isdigit():
                num_features.append('[Num]1')
            else:
                num_features.append('[Num]0')
        return num_features

    def add_location_feature_recursive(self, sentence, max_lenght, startidx, feature):
        if max_lenght > len(sentence) - len(feature):
            max_lenght = len(sentence) - len(feature)
        if max_lenght <= 0:
            return feature
        word = ''
        for idx in range(max_lenght):
            word += " " + sentence[startidx + idx]
        word = word.replace('_', ' ').strip().lower()
        for count in range(max_lenght):
            if word.lower() in self.loc_dict:
                num_word = max_lenght - count
                for _ in range(num_word):
                    feature.append('[LOC]1')
                feature = self.add_location_feature_recursive(sentence, max_lenght, startidx + num_word, feature)
                break
            else:
                word = word.rsplit(' ', 1)[0]
        if len(feature) == startidx:
            feature.append('[LOC]0')
            feature = self.add_location_feature_recursive(sentence, max_lenght, startidx + 1, feature)
        return feature

    def add_organization_feature_recursive(self, sentence, max_lenght, startidx, feature):
        if max_lenght > len(sentence) - len(feature):
            max_lenght = len(sentence) - len(feature)
        if max_lenght <= 0:
            return feature
        word = ''
        for idx in range(max_lenght):
            word += " " + sentence[startidx + idx]
        word = word.replace('_', ' ').strip().lower()
        for count in range(max_lenght):
            if word.lower() in self.org_dict:
                num_word = max_lenght - count
                for _ in range(num_word):
                    feature.append('[ORG]1')
                feature = self.add_organization_feature_recursive(sentence, max_lenght, startidx + num_word, feature)
                break
            else:
                word = word.rsplit(' ', 1)[0]
        if len(feature) == startidx:
            feature.append('[ORG]0')
            feature = self.add_organization_feature_recursive(sentence, max_lenght, startidx + 1, feature)
        return feature

    def add_person_feature_recursive(self, sentence, max_lenght, startidx, feature):
        if max_lenght > len(sentence) - len(feature):
            max_lenght = len(sentence) - len(feature)
        if max_lenght <= 0:
            return feature
        word = ''
        for idx in range(max_lenght):
            word += " " + sentence[startidx + idx]
        word = word.replace('_', ' ').strip().lower()
        for count in range(max_lenght):
            if word.lower() in self.per_dict:
                num_word = max_lenght - count
                for _ in range(num_word):
                    feature.append('[PER]1')
                feature = self.add_person_feature_recursive(sentence, max_lenght, startidx + num_word, feature)
                break
            else:
                word = word.rsplit(' ', 1)[0]
        if len(feature) == startidx:
            feature.append('[PER]0')
            feature = self.add_person_feature_recursive(sentence, max_lenght, startidx + 1, feature)
        return feature

    def add_person_position_feature_recursive(self, sentence, max_lenght, startidx, feature):
        if max_lenght > len(sentence) - len(feature):
            max_lenght = len(sentence) - len(feature)
        if max_lenght <= 0:
            return feature
        word = ''
        for idx in range(max_lenght):
            word += " " + sentence[startidx + idx]
        word = word.replace('_', ' ').strip().lower()
        for count in range(max_lenght):
            if word.lower() in self.ppos_dict:
                num_word = max_lenght - count
                for _ in range(num_word):
                    feature.append('[PPOS]1')
                feature = self.add_person_position_feature_recursive(sentence, max_lenght, startidx + num_word, feature)
                break
            else:
                word = word.rsplit(' ', 1)[0]
        if len(feature) == startidx:
            feature.append('[PPOS]0')
            feature = self.add_person_position_feature_recursive(sentence, max_lenght, startidx + 1, feature)
        return feature

    @staticmethod
    def recover_feature(orginal_sentence, feature, feat_type):
        new_feature = []
        for words in orginal_sentence:
            word = words.split("_")
            if (feat_type + '0') in feature[0:len(word)]:
                new_feature.append(feat_type + '0')
            else:
                new_feature.append(feat_type + '1')
            del feature[0:len(word)]
        return new_feature

    def add_dict_feature(self, sentence: list, dict_type: str = '[LOC]') -> list:
        features = []
        dict_features = []
        for word in sentence:
            dict_features.extend(word.replace('_', ' ').split(' '))
        if dict_type == '[LOC]':
            features = self.add_location_feature_recursive(dict_features, self.loc_max_lenght, 0, features)
        elif dict_type == '[ORG]':
            features = self.add_organization_feature_recursive(dict_features, self.org_max_lenght, 0, features)
        elif dict_type == '[PER]':
            features = self.add_person_feature_recursive(dict_features, self.per_max_lenght, 0, features)
        elif dict_type == '[PPOS]':
            features = self.add_person_position_feature_recursive(dict_features, self.ppos_max_lenght, 0, features)
        dict_features = self.recover_feature(sentence, features, dict_type)
        return dict_features

    def extract_feature(self, sentence: str or list, pos_tags: list = None, ner_labels: list = None, format=None):
        features = []
        result = []
        if "pos" in self.feature_types:
            sentence, pos_features, ner_labels = self.wseg_and_add_pos_tag_feature(sentence, pos_tags, ner_labels)
            features.append(pos_features)
        else:
            sentence, ner_labels = self.word_segment(sentence, ner_labels)
        if "cf" in self.feature_types:
            features.append(self.add_case_feature(sentence))
        if "sc" in self.feature_types:
            features.append(self.add_sequence_case_feature(sentence))
        if "fw" in self.feature_types:
            features.append(self.add_fisrt_word_feature(sentence))
        if "qb" in self.feature_types:
            features.append(self.add_quotes_brackets_feature(sentence))
        if "num" in self.feature_types:
            features.append(self.add_number_feature(sentence))
        if "loc" in self.feature_types:
            features.append(self.add_dict_feature(sentence, '[LOC]'))
        if "org" in self.feature_types:
            features.append(self.add_dict_feature(sentence, '[ORG]'))
        if "per" in self.feature_types:
            features.append(self.add_dict_feature(sentence, '[PER]'))
        if "ppos" in self.feature_types:
            features.append(self.add_dict_feature(sentence, '[PPOS]'))
        if format == "text":
            for idx, token in enumerate(sentence):
                example = str(sentence[idx])
                for feature in range(len(features)):
                    example = f"{example}\t{features[feature][idx]}"
                if ner_labels is not None:
                    example += f"\t{ner_labels[idx]}"
                result.append(example)
        else:
            feats = []
            for idx, _ in enumerate(sentence):
                feat = []
                for feature in features:
                    k, v = feature[idx].split("]")
                    feat.append((f"{k}]", v))
                feats.append(feat)
            result = (sentence, feats)
        return result


if __name__ == "__main__":
    print("Build Extractor ...")
    fe = FeatureExtractor(dict_dir='resources/features')
    text = "Chỉ	riêng xã Cương Gián	(Hà Tĩnh) đã có 10 thuyền viên tử nạn giữa trùng dương bao la."
    print(f"Input text: {text}")
    print(fe.extract_feature(text))
