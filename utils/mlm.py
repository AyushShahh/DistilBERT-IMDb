import torch
from transformers import DistilBertTokenizer

class MaskedLanguageModeling():
    def __init__(self, tokenizer=None, mlm_probability=0.15):
        self.tokenizer = tokenizer if tokenizer is not None else DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.mlm_probability = mlm_probability
        self.mask_token_id = self.tokenizer.mask_token_id
        self.vocab_size = self.tokenizer.vocab_size
        self.DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __call__(self, inputs):
        input_ids = inputs["input_ids"].clone()
        labels = input_ids.clone()

        # Select tokens to mask
        probability_matrix = torch.full(labels.shape, self.mlm_probability, device=self.DEVICE)
        special_tokens_mask = torch.tensor(
            [
                self.tokenizer.get_special_tokens_mask(seq, already_has_special_tokens=True)
                for seq in input_ids.tolist()
            ],
            dtype=torch.bool,
            device=self.DEVICE
        )
        probability_matrix[special_tokens_mask] = 0.0
        probability_matrix[inputs["attention_mask"] == 0] = 0.0
        masked_indices = torch.bernoulli(probability_matrix).bool()

        # Non masked labels are set to -100
        labels[~masked_indices] = -100

        # 80%, replace with [MASK]
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8, device=self.DEVICE)).bool()
            & masked_indices
        )
        input_ids[indices_replaced] = self.mask_token_id

        # 10%, random token
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5, device=self.DEVICE)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            self.vocab_size,
            labels.shape,
            dtype=torch.long,
            device=self.DEVICE
        )
        input_ids[indices_random] = random_words[indices_random]

        return {
            "input_ids": input_ids,
            "attention_mask": inputs["attention_mask"],
            "labels": labels
        }
