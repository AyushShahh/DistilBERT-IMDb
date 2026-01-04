import torch
from torch.amp import autocast
import random
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import json
from tqdm import tqdm

def setup(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.set_float32_matmul_precision('high')
        torch.backends.cudnn.benchmark = False

def plot_curves(history, root):
    _, axes = plt.subplots(1, 2, figsize=(12, 4))

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
    plt.savefig(f'{root}/training_curves.png', dpi=150)
    plt.show()

def calculate_and_save_metrics(all_preds, all_labels, all_probs, ROOT):
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
    with open(f'{ROOT}/test_metrics.json', 'w') as f:
        json.dump(test_metrics, f, indent=2)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
    _, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'{ROOT}/confusion_matrix.png', dpi=150)
    plt.show()

def test_model(model, path, testloader, DEVICE):
    model.load_state_dict(torch.load(path))
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
    
    return all_preds, all_labels, all_probs
