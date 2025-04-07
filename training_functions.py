import torch
import math


def training_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    epoch_loss = 0
    correct = 0
    n = 0
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        loss = criterion(output, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = output.max(1)
        correct += predicted.eq(y).sum().item()
        n += y.size(0)

    avg_loss = epoch_loss / n
    accuracy = 100 * correct / n
    return accuracy, avg_loss


def evaluate(model, dataloader, criterion, device):
    correct = 0
    n = 0
    model.eval()
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            output = model(x)
            _, predicted = torch.max(output, 1)

            loss = criterion(output, y)
            batch_loss = loss.item()
            n += y.size(0)
            correct += predicted.eq(y).sum().item()

    avg_loss = batch_loss / n
    accuracy = 100 * correct / n
    return accuracy, avg_loss


def train(epochs, model, train_dataloader, val_dataloader, optimizer, criterion, device, model_path,
          tolerance=math.inf):
    train_accuracy_list, train_loss_list = [], []
    val_accuracy_list, val_loss_list = [], []
    best_loss = float('inf')
    last_save = 0
    epochs_without_improvement = 0

    for epoch in range(epochs):
        train_accuracy, train_avg_loss = training_epoch(model, train_dataloader, optimizer, criterion, device)
        train_accuracy_list.append(train_accuracy)
        train_loss_list.append(train_avg_loss)
        print(f"epoch: {epoch + 1}, training loss: {train_avg_loss}, training accuracy: {train_accuracy}")

        val_accuracy, val_avg_loss = evaluate(model, val_dataloader, criterion, device)
        val_accuracy_list.append(val_accuracy)
        val_loss_list.append(val_avg_loss)
        print(f"epoch: {epoch + 1}, validation loss: {val_avg_loss}, validation accuracy: {val_accuracy}")

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
        "loss_val": val_loss_list,
        "accuracy_val": val_accuracy_list,
        "last_save": last_save
    }
    return history
