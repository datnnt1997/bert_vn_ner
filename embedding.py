import torch
import torch.nn as nn
import numpy as np


class WordRep(nn.Module):
    def __init__(self, data):
        super(WordRep, self).__init__()
        for idx in range(self.feature_num):
            self.feature_embeddings.append(nn.Embedding(data.feature_alphabets[idx].size(), self.feature_embedding_dims[idx]))
        for idx in range(self.feature_num):
            if data.pretrain_feature_embeddings[idx] is not None:
                self.feature_embeddings[idx].weight.data.copy_(torch.from_numpy(data.pretrain_feature_embeddings[idx]))
            else:
                self.feature_embeddings[idx].weight.data.copy_(torch.from_numpy(
                    self.random_embedding(data.feature_alphabets[idx].size(), self.feature_embedding_dims[idx])))
        if self.gpu:
            self.drop = self.drop.cuda()
            for idx in range(self.feature_num):
                self.feature_embeddings[idx] = self.feature_embeddings[idx].cuda()

    def forward(self, bert_embed, features):
        token_embed = [bert_embed]
        for idx in range(self.feature_num):
            token_embed.append(self.feature_embeddings[idx](features[idx]))
        word_embs = torch.cat(token_embed, 2)