import torch
from torch import nn
from transformers import AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput


class ModelForClassification(nn.Module):
    def __init__(self, model_name: str, num_labels: int):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        hidden_size = self.model.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_values: torch.Tensor) -> SequenceClassifierOutput:
        outputs = self.model(input_values)
        pooled = outputs.last_hidden_state.mean(dim=1)
        logits = self.classifier(self.dropout(pooled))
        return SequenceClassifierOutput(logits=logits)
