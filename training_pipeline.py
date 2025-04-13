import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from serialization import save
from training_functions import train, evaluate
import math
import random
import numpy as np


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

def repeat_training(n, init_model, lr, model_path, history_path, epochs, train_dataloader, val_dataloader, test_dataloader, device, dropout=False, betas=(0.9, 0.999), weight_decay=0, tolerance=math.inf):
    for i in range(n):
        if not dropout:
            model = init_model()
        else:
            model = init_model(dropout)

        model.to(device)

        print(f"training iteration: {i+1} of {n}")
        criterion = nn.CrossEntropyLoss()
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
            best_model = init_model(dropout)

        best_model.to(device)

        best_model.load_state_dict(torch.load(model_path_idx, weights_only=True))
        test_accuracy, test_avg_loss, test_bal_acc = evaluate(best_model, test_dataloader, criterion, device)
        print(f"test loss: {test_avg_loss}, test accuracy: {test_accuracy}, test balanced accuracy: {test_bal_acc}")

        training_history["accuracy_test"] = test_accuracy
        training_history["loss_test"] = test_avg_loss
        training_history["balanced_accuracy_test"] = test_bal_acc

        save(training_history, history_path_idx)
        print("training history saved\n")
