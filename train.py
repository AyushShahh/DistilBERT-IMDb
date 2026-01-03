import os
import numpy as np
import torch
from torch.utils.data import random_split
from utils.dataset import IMDbReviewsDataLoader, IMDbReviews
from utils.architecture import DistilBERTClassifier
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import json


def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 64

    trainloader = IMDbReviewsDataLoader(
        split='train',
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
    )

    test_dataset = IMDbReviews(split='test')
    val_size = int(0.2 * len(test_dataset))
    test_size = len(test_dataset) - val_size
    generator = torch.Generator().manual_seed(23)
    val_dataset, test_dataset = random_split(test_dataset, [val_size, test_size], generator=generator)

    valloader = IMDbReviewsDataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    testloader = IMDbReviewsDataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    EPOCHS = 3

    num_train_steps = len(trainloader) * EPOCHS
    warmup_steps = int(0.1 * num_train_steps)

    model = DistilBERTClassifier().to(DEVICE)
    optimizer = torch.optim.AdamW([
        {'params': model.distilbert.parameters(), 'lr': 1e-5},
        {'params': model.classifier.parameters(), 'lr': 5e-5},
    ])

    # Linear decay with warmup
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(1e-8, float(num_train_steps - current_step) / float(max(1, num_train_steps - warmup_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    criterion = torch.nn.CrossEntropyLoss()

    scaler = GradScaler()
    best_val_acc = 0.0

    # Loss tracking
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'avg_train_loss': [],
        'avg_val_loss': [],
        'avg_train_acc': [],
        'avg_val_acc': []
    }

    for epoch in range(EPOCHS):
        # Training
        model.train()
        epoch_loss = 0.0
        train_correct = 0
        train_total = 0

        progress_bar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False, unit="batch")
        for batch in progress_bar:
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            with autocast(device_type=DEVICE):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct = (preds == labels).sum().item()
            train_correct += correct
            train_total += labels.size(0)

            history['train_loss'].append(loss.item())
            history['train_acc'].append(correct / labels.size(0))

            progress_bar.set_postfix(loss=loss.item(), acc=correct / labels.size(0))

        avg_train_loss = epoch_loss / len(trainloader)
        train_acc = train_correct / train_total

        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(valloader, desc=f"Epoch {epoch+1}/{EPOCHS} [Val]", leave=False, unit="batch")
        with torch.no_grad():
            for batch in loop:
                input_ids = batch['input_ids'].to(DEVICE)
                attention_mask = batch['attention_mask'].to(DEVICE)
                labels = batch['labels'].to(DEVICE)

                with autocast(device_type=DEVICE):
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, labels)

                val_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                c = (preds == labels).sum().item()
                correct += c
                total += labels.size(0)

                loop.set_postfix(loss=loss.item(), acc=c / labels.size(0))

                history['val_loss'].append(loss.item())
                history['val_acc'].append(c / labels.size(0))

        avg_val_loss = val_loss / len(valloader)
        val_acc = correct / total

        # Save to history
        history['avg_train_loss'].append(avg_train_loss)
        history['avg_train_acc'].append(train_acc)
        history['avg_val_loss'].append(avg_val_loss)
        history['avg_val_acc'].append(val_acc)

        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('assets', exist_ok=True)
            torch.save(model.state_dict(), 'assets/best_model.pt')

    # Save loss history to JSON
    with open('assets/training_history.json', 'w') as f:
        json.dump(history, f, indent=2)

    # Plot and save loss curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(np.linspace(0, 3, len(history['train_loss'])), history['train_loss'], label='Train Loss', linewidth=0.5, alpha=0.8)
    axes[0].plot(np.linspace(0, 3, len(history['val_loss'])), history['val_loss'], label='Val Loss', linewidth=0.5, alpha=0.8)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curve')
    axes[0].legend()

    axes[1].plot(np.linspace(0, 3, len(history['train_acc'])), history['train_acc'], label='Train Acc', linewidth=0.5, alpha=0.8)
    axes[1].plot(np.linspace(0, 3, len(history['val_acc'])), history['val_acc'], label='Val Acc', linewidth=0.5, alpha=0.8)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curve')
    axes[1].legend()

    plt.tight_layout()
    plt.savefig('assets/training_curves.png', dpi=150)
    plt.show()

    # Final evaluation on test set
    model.load_state_dict(torch.load('assets/best_model.pt'))
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in tqdm(testloader, desc="Testing", unit="batch"):
            input_ids = batch['input_ids'].to(DEVICE)
            attention_mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)

            with autocast(device_type=DEVICE):
                logits = model(input_ids, attention_mask)

            probs = torch.softmax(logits, dim=1)[:, 1]
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Calculate metrics
    test_acc = sum(p == l for p, l in zip(all_preds, all_labels)) / len(all_labels)
    test_f1 = f1_score(all_labels, all_preds)
    test_precision = precision_score(all_labels, all_preds)
    test_recall = recall_score(all_labels, all_preds)
    test_roc_auc = roc_auc_score(all_labels, all_probs)

    print(f"Test Results:")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"F1 Score: {test_f1:.4f}")
    print(f"Precision {test_precision:.4f}")
    print(f"Recall: {test_recall:.4f}")
    print(f"ROC-AUC: {test_roc_auc:.4f}")

    # Save test metrics to JSON
    test_metrics = {
        'accuracy': test_acc,
        'f1_score': test_f1,
        'precision': test_precision,
        'recall': test_recall,
        'roc_auc': test_roc_auc
    }
    with open('assets/test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('assets/confusion_matrix.png', dpi=150)
    plt.show()


if __name__ == "__main__":
    main()