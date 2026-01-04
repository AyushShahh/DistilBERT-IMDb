import torch
from transformers import DistilBertConfig, DistilBertModel


class DistilBERTClassifier(torch.nn.Module):
    def __init__(self, num_labels=2):
        super(DistilBERTClassifier, self).__init__()

        config = DistilBertConfig.from_pretrained("ayushshah/distilbert-base-uncased-imdb-dapt")
        self.distilbert = DistilBertModel(config)
        self.classifier = torch.nn.Linear(self.distilbert.config.hidden_size, num_labels)
        self.dropout = torch.nn.Dropout(0.1)

    def forward(self, input_ids, attention_mask):
        hidden_states = self.distilbert(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        pooled_output = hidden_states[:, 0, :]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits

def infer_reviews(reviews, model, tokenizer):
    is_single = isinstance(reviews, str)
    if is_single:
        reviews = [reviews]
    
    inputs = tokenizer(reviews, return_tensors="pt", truncation=True, padding=True, max_length=512).to('cuda' if torch.cuda.is_available() else 'cpu')
    
    with torch.no_grad():
        logits = model(inputs['input_ids'], inputs['attention_mask'])
        probs = torch.softmax(logits, dim=1)
        predicted_classes = torch.argmax(probs, dim=1)
        confidences = probs[range(len(predicted_classes)), predicted_classes]
    
    results = []
    for pred_class, confidence in zip(predicted_classes.tolist(), confidences.tolist()):
        class_label = "Positive" if pred_class == 1 else "Negative"
        results.append((pred_class, class_label, confidence))
    
    return results[0] if is_single else results