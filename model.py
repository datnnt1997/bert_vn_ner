import torch.nn as nn

from transformers.tokenization_bert import BertTokenizer
from transformers.modeling_bert import BertModel, BertPreTrainedModel, BertConfig


class BertNer(BertPreTrainedModel):
    def __init__(self, config):
        super(BertNer, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)
        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)
        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        return outputs  # (loss), scores, (hidden_states), (attentions)

    def calculate_loss(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                       attention_mask_label=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            head_mask=None)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        loss_function = nn.CrossEntropyLoss()
        # Only keep active parts of the loss
        if attention_mask_label is not None:
            active_loss = attention_mask_label.view(-1) == 1
            active_logits = logits.view(-1, self.num_labels)[active_loss]
            active_labels = labels.view(-1)[active_loss]
            loss = loss_function(active_logits, active_labels)
        else:
            loss = loss_function(logits.view(-1, self.num_labels), labels.view(-1))
        outputs = (loss,) + outputs
        return outputs  # (loss), scores, (hidden_states), (attentions)


def modelbuilder(model_name_or_path, num_labels):
    config = BertConfig.from_pretrained(model_name_or_path, num_labels=num_labels)
    model = BertNer.from_pretrained(model_name_or_path, config=config)
    return config, model


if __name__ == "__main__":
    config, tokenizer, model = modelbuilder("bert-base-multilingual-uncased")
    print(model)