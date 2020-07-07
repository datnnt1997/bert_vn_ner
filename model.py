import torch
import torch.nn as nn

from transformers.modeling_bert import BertModel, BertPreTrainedModel, BertConfig


class BertNer(BertPreTrainedModel):
    def __init__(self, config):
        super(BertNer, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, attention_mask_label=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            head_mask=None)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)

        logits = self.classifier(sequence_output)

        active_loss = attention_mask_label.view(-1) == 1
        active_logits = logits.view(-1, self.num_labels)[active_loss]

        return active_logits

    def calculate_loss(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                       attention_mask_label=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            head_mask=None)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        loss_function = nn.CrossEntropyLoss()
        # Only keep active parts of the loss
        active_loss = attention_mask_label.view(-1) == 1
        active_logits = logits.view(-1, self.num_labels)[active_loss]
        active_labels = labels.view(-1)[active_loss]
        loss = loss_function(active_logits, active_labels)
        outputs = (loss, active_logits)
        return outputs  # (loss), logits


def modelbuilder(model_name_or_path, num_labels):
    config = BertConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
    model = BertNer.from_pretrained(model_name_or_path, config=config)
    return config, model

def model_builder_from_pretrained(model_name_or_path, num_labels, pre_train_path):
    config = BertConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
    model = BertNer.from_pretrained(model_name_or_path, config=config)
    model.load_state_dict(torch.load(pre_train_path+"/vner_model.bin", map_location='cpu'))
    model.eval()
    return config, model


if __name__ == "__main__":
    config, tokenizer, model = modelbuilder("bert-base-multilingual-uncased")
    print(model)