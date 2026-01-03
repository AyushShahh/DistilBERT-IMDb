import torch.nn as nn
from transformers import DistilBertModel
from huggingface_hub import PyTorchModelHubMixin


class DistilBERTClassifier(nn.Module, PyTorchModelHubMixin):
    def __init__(self, distilbert_model="ayushshah/distilbert-base-uncased-imdb-dapt", num_labels=2):
        super(DistilBERTClassifier, self).__init__()

        self.distilbert = DistilBertModel.from_pretrained(distilbert_model)
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        hidden_states = self.distilbert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled_output = hidden_states[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits