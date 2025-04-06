import torch
from torch.utils.data import DataLoader
from custom_dataset import AudioDataset


def main():
    print("starting processing...")
    data_path = "data/train/audio"
    dataset = AudioDataset(data_path, set_type=AudioDataset.TRAIN)
    print("creating dataloader...")
    train_loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    mean = count_mean(train_loader, device)
    print(mean)

    std = count_std(train_loader, device, mean)
    print(std)

def count_mean(train_loader, device):
    sum = torch.tensor(0.0, device=device)
    num_pixels = torch.tensor(0.0, device=device)

    for log_spec, _ in train_loader:
        print("new batch processing...")
        log_spec = log_spec.to(device)
        log_spec = log_spec.view(log_spec.size(0), -1)

        sum += log_spec.sum()
        num_pixels += log_spec.numel()

    mean = sum / num_pixels
    return mean

def count_std(train_loader, device, mean):
    sum = torch.tensor(0.0, device=device)
    num_pixels = torch.tensor(0.0, device=device)

    for log_spec, _ in train_loader:
        print("new batch processing...")
        log_spec = log_spec.to(device)
        log_spec = log_spec.view(log_spec.size(0), -1)

        elements = torch.pow(log_spec - mean, 2)

        sum += elements.sum()
        num_pixels += log_spec.numel()

    std = torch.sqrt(sum / num_pixels-1)
    return std


if __name__ == "__main__":
    main()
