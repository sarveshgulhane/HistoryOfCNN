import os
import torch
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_epoch(model, data_loader, optimizer, loss_fn):
    """Train for one epoch. Signature: (model, data_loader, optimizer, loss_fn)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(data_loader, desc="Train", leave=False)

    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        # avoid division by zero if dataset empty
        pbar.set_postfix(
            loss=(running_loss / total) if total else 0.0,
            acc=(correct / total) if total else 0.0,
        )

    avg_loss = (running_loss / total) if total else 0.0
    acc = (correct / total) if total else 0.0
    return avg_loss, acc


def eval_epoch(model, data_loader, loss_fn):
    """Evaluate for one epoch. Signature: (model, data_loader, loss_fn)."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(data_loader, desc="Eval", leave=False)

    with torch.no_grad():
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            pbar.set_postfix(
                loss=(running_loss / total) if total else 0.0,
                acc=(correct / total) if total else 0.0,
            )

    avg_loss = (running_loss / total) if total else 0.0
    acc = (correct / total) if total else 0.0
    return avg_loss, acc


def train_and_save_model(
    model, epochs, optimizer, loss_fn, data_loader, save_path
):
    train_loader, test_loader = data_loader

    model = model.to(device)

    best_acc = 0.0
    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, loss_fn
        )
        val_loss, val_acc = eval_epoch(model, test_loader, loss_fn)
        # scheduler.step()  # if using a scheduler

        print(f"  Train loss: {train_loss:.4f}  Train acc: {train_acc:.4f}")
        print(f"  Val   loss: {val_loss:.4f}  Val   acc: {val_acc:.4f}")

        # save best
        if val_acc > best_acc:
            best_acc = val_acc
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_acc": val_acc,
                },
                save_path,
            )
            print(f"  Saved best model (val_acc={val_acc:.4f}) -> {save_path}")

    print("Done. Best val acc:", best_acc)
