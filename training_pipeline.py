import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from serialization import save, load
from training_functions import train, evaluate
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def worker_init_fn(worker_id):
    seed = torch.initial_seed() % 2 ** 32
    set_seed(seed)

def add_prefix_to_path(path, prefix):
    dirpath, filename = os.path.split(path)
    name, ext = os.path.splitext(filename)
    file = f"{name}_{prefix}{ext}"
    new_path = os.path.join(dirpath, file)
    return new_path

def repeat_training(n, init_model, lr, model_path, history_path, epochs, train_dataloader, val_dataloader, test_dataloader, device, dropout=False, betas=(0.9, 0.999), weight_decay=0, tolerance=math.inf,
                    label_smoothing=0):
    for i in range(n):
        if not dropout:
            model = init_model()
        else:
            model = init_model(dropout=dropout)

        model.to(device)

        print(f"training iteration: {i+1} of {n}")
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        # TODO enable modifying optimizer (it must be initialized after every training)
        optimizer = optim.Adam(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)

        model_path_idx = add_prefix_to_path(model_path, i+1)
        history_path_idx = add_prefix_to_path(history_path, i+1)

        start_time = time.time()
        print("starting training...")
        training_history = train(epochs, model, train_dataloader, val_dataloader, optimizer, criterion, device,
                                 model_path_idx, tolerance)
        print("training finished\n")
        print(training_history)
        end_time = time.time()
        print(f"training time: {end_time - start_time}\n")

        print("evaluating model...")
        if not dropout:
            best_model = init_model()
        else:
            best_model = init_model(dropout=dropout)

        best_model.to(device)

        best_model.load_state_dict(torch.load(model_path_idx, weights_only=True))
        test_accuracy, test_avg_loss, test_bal_acc = evaluate(best_model, test_dataloader, criterion, device)
        print(f"test loss: {test_avg_loss}, test accuracy: {test_accuracy}, test balanced accuracy: {test_bal_acc}")

        training_history["accuracy_test"] = test_accuracy
        training_history["loss_test"] = test_avg_loss
        training_history["balanced_accuracy_test"] = test_bal_acc

        save(training_history, history_path_idx)
        print("training history saved\n")

def plot_results(history_dir, x_values, x_label, use_balanced_accuracy=False):
    data = []
    # for history_dir in history_dir:
    for dir in os.listdir(history_dir):
        accuracy_results = []
        balanced_accuracy_results = []
        dir_path = os.path.join(history_dir, dir)
        for file_name in os.listdir(dir_path):
            if file_name.endswith(".pkl"):
                history_path = os.path.join(dir_path, file_name)
                history = load(history_path)

                accuracy_test = history["accuracy_test"]
                accuracy_results.append(accuracy_test)

                balanced_accuracy_test = history["balanced_accuracy_test"]
                balanced_accuracy_results.append(balanced_accuracy_test)

        data.append(balanced_accuracy_results if use_balanced_accuracy else accuracy_results)

    y_label = "test balanced accuracy" if use_balanced_accuracy else "test accurac"
    plot_data = []
    for i in range(len(x_values)):
        param = x_values[i]
        results = data[i]
        for res in results:
            plot_data.append({x_label: param, y_label: res})

    df = pd.DataFrame(plot_data)

    plt.figure(figsize=(8, 5))
    sns.boxplot(x=x_label, y=y_label, data=df)
    plt.show()
