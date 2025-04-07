import torch
from torch.utils.data import DataLoader
from custom_dataset import SpectrogramDataset


def main():
    print("starting processing...")
    data_path = "data/train/audio_transformed"
    dataset = SpectrogramDataset(data_path, set_type=SpectrogramDataset.TRAIN)
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

    print("\nComputed results:")
    print(f"mean: {mean}")
    print(f"std: {std}")

    # Computed results:
    # mean: -10.408944129943848
    # std: 5.073166847229004

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
