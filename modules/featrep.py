from commons import Feature

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureRep(nn.Module):
    def __init__(self, feature: Feature, device: str = 'cpu'):
        super(FeatureRep, self).__init__()
        self.feature_embeddings = {}
        self.feature_num = feature.num_of_feature
        self.feature_dim = feature.feature_emb_dim
        for feat_key in feature.feature_keys:
            feat = feature.feature_infos[feat_key]
            self.feature_embeddings[feat_key] = nn.Embedding(len(feat['label']) + 1, feat['dim'], padding_idx=0)

        if feature.one_hot_emb:
            for feat_key in feature.feature_keys:
                feat = feature.feature_infos[feat_key]
                one_hot_weight = F.one_hot(torch.arange(feat['dim']))
                one_hot_weight = torch.cat([torch.zeros((1, feat['dim']), dtype=torch.int64), one_hot_weight], dim=0)
                self.feature_embeddings[feat_key].weight.data.copy_(one_hot_weight)
                self.feature_embeddings[feat_key].weight.requires_grad = False
        else:
            for feat_key in feature.feature_keys:
                feat = feature.feature_infos[feat_key]
                self.feature_embeddings[feat_key].weight.data.copy_(torch.from_numpy(
                    self.random_embedding(len(feat['label'])+1, feat['dim'])))


        if device.type == 'cuda':
            for feat_key in feature.feature_keys:
                self.feature_embeddings[feat_key] = self.feature_embeddings[feat_key].cuda()

    @staticmethod
    def random_embedding(vocab_size, embedding_dim):
        pretrain_emb = np.empty([vocab_size, embedding_dim])
        scale = np.sqrt(3.0 / embedding_dim)
        for index in range(vocab_size):
            pretrain_emb[index, :] = np.random.uniform(-scale, scale, [1, embedding_dim])
        return pretrain_emb

    def forward(self, features: list):
        feat_reps = []
        for feat_key, feat_value in features.items():
            feat_reps.append(self.feature_embeddings[feat_key](feat_value))
        feat_embs = torch.cat(feat_reps, dim=-1)
        return feat_embs


if __name__ == "__main__":
    feature = Feature("resources/feature_config.json", True)
    featrep = FeatureRep(feature)
    print(featrep)
    sample_input = []
    for feat_key in feature.feature_keys:
        vocab_size = len(feature.feature_infos[feat_key]["label"])
        sample_input.append(torch.randint(low=0, high=vocab_size, size=(1, 5), dtype=torch.long))
    outs = featrep(sample_input)
    print(outs.shape)