import torch
from torch import nn
from transformers import (
    AutoConfig,
    AutoModel,
    WavLMModel,
    WavLMPreTrainedModel,
)
from transformers.modeling_outputs import TokenClassifierOutput, SequenceClassifierOutput


class ModelForClassification(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()

        self.num_labels = num_labels

        self.model = AutoModel.from_pretrained(model_name)
        hidden_size = self.model.config.hidden_size

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_values: torch.Tensor, labels=None):
        outputs = self.model(input_values)

        hidden_states = outputs.last_hidden_state


        pooled = hidden_states.mean(dim=1)

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return TokenClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class WavLMForSequenceClassification2(WavLMPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)

        self.wavlm = WavLMModel(config)

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.post_init()

    def forward(
        self,
        input_values: torch.Tensor,
        attention_mask=None,
        labels=None,
        return_dict=True,
    ):
        outputs = self.wavlm(
            input_values,
            attention_mask=attention_mask,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)
            hidden_states = hidden_states * mask
            pooled = hidden_states.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)
        else:
            pooled = hidden_states.mean(dim=1)

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        if not return_dict:
            return (loss, logits) if loss is not None else (logits,)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


if __name__ == "__main__":
    config = AutoConfig.from_pretrained("microsoft/wavlm-base")
    model = WavLMForSequenceClassification2(config)