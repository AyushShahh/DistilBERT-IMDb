import torch
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizer, DataCollatorWithPadding
from datasets import load_dataset


class IMDbReviews(Dataset):
    def __init__(self, split, tokenizer_name='distilbert-base-uncased', max_length=512, encode=True):
        self.split = split
        self.encode = encode
        self.data = load_dataset("stanfordnlp/imdb", split=split)
        if self.encode:
            self.tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_name)
            self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if not self.encode:
            return self.data[idx]

        review, label = self.data[idx]['text'], self.data[idx]['label']
        encoding = self.tokenizer(
            review,
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item


class IMDbReviewsDataLoader(DataLoader):
    def __init__(self, split, dataset=None, batch_size=8, shuffle=True, num_workers=2, pin_memory=True, persistent_workers=False, **kwargs):
        if dataset is None:
            self.dataset = IMDbReviews(split=split)
        else:
            self.dataset = dataset
        self.collator = DataCollatorWithPadding(
            tokenizer=self.dataset.tokenizer,
            pad_to_multiple_of=8,
            return_tensors="pt"
        )
        super(IMDbReviewsDataLoader, self).__init__(
            self.dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self.collator,
            persistent_workers=persistent_workers,
            **kwargs
        )