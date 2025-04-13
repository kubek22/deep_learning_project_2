import torch
from sklearn.metrics import balanced_accuracy_score


def training_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    correct = 0
    n = 0
    all_targets = []
    all_predictions = []
    for x, y in dataloader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        output = model(x)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = output.max(1)
        correct += predicted.eq(y).sum().item()
        n += y.size(0)

        all_targets.extend(y.cpu().numpy())
        all_predictions.extend(predicted.cpu().numpy())

    avg_loss = epoch_loss / n
    accuracy = 100 * correct / n
    balanced_acc = balanced_accuracy_score(all_targets, all_predictions) * 100
    return accuracy, avg_loss, balanced_acc

def evaluate(model, dataloader, criterion, device):
    total_loss = 0
    correct = 0
    n = 0
    all_targets = []
    all_predictions = []

    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            output = model(x)
            _, predicted = torch.max(output, 1)

            loss = criterion(output, y)
            total_loss += loss.item()
            n += y.size(0)
            correct += predicted.eq(y).sum().item()

            all_targets.extend(y.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    avg_loss = total_loss / n
    accuracy = 100 * correct / n
    balanced_acc = balanced_accuracy_score(all_targets, all_predictions) * 100
    return accuracy, avg_loss, balanced_acc

def train(epochs, model, train_dataloader, val_dataloader, optimizer, criterion, device, model_path,
          tolerance=torch.inf):
    train_accuracy_list, train_loss_list, train_balanced_acc_list = [], [], []
    val_accuracy_list, val_loss_list, val_balanced_acc_list = [], [], []
    best_loss = float('inf')
    last_save = 0
    epochs_without_improvement = 0

    for epoch in range(epochs):
        train_accuracy, train_avg_loss, train_balanced_acc = training_epoch(model, train_dataloader, optimizer, criterion, device)
        train_accuracy_list.append(train_accuracy)
        train_loss_list.append(train_avg_loss)
        train_balanced_acc_list.append(train_balanced_acc)
        print(f"epoch: {epoch + 1}, training loss: {train_avg_loss}, training accuracy: {train_accuracy}, training balanced accuracy: {train_balanced_acc}")

        val_accuracy, val_avg_loss, val_balanced_acc = evaluate(model, val_dataloader, criterion, device)
        val_accuracy_list.append(val_accuracy)
        val_loss_list.append(val_avg_loss)
        val_balanced_acc_list.append(val_balanced_acc)
        print(f"epoch: {epoch + 1}, validation loss: {val_avg_loss}, validation accuracy: {val_accuracy}, validation balanced accuracy: {val_balanced_acc}")

        if val_avg_loss < best_loss:
            best_loss = val_avg_loss
            torch.save(model.state_dict(), model_path)
            last_save = epoch + 1
            epochs_without_improvement = 0
            print("model saved")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement > tolerance:
                print(f"Training stopped. Tolerance {tolerance} exceeded")
                break
        print()

    history = {
        "loss_train": train_loss_list,
        "accuracy_train": train_accuracy_list,
        "balanced_accuracy_train": train_balanced_acc_list,
        "loss_val": val_loss_list,
        "accuracy_val": val_accuracy_list,
        "last_save": last_save
    }
    return history
