import numpy as np
import torch
from torch.utils.data import DataLoader
from custom_dataset import SpectrogramDataset, AudioDataset


def main():
    # main function is required for multiprocessing
    dataset_test()

def dataset_test():
    data_path = "data/train/audio"
    dataset = AudioDataset(data_path, set_type=AudioDataset.TRAIN)
    loader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=4,
        pin_memory=False,
    )

    labels = []
    for x, y in dataset.samples:
        labels.append(y)
    labels = np.array(labels)

    values, counts = np.unique(labels, return_counts=True)
    value_counts = dict(zip(values, counts))
    print(value_counts)
    print(np.sum(counts))

    print(dataset.samples[0])
    print(len(dataset.samples))

    wave = dataset.__getitem__(0)[0]
    print(wave)
    print(wave.shape)
    print(torch.max(wave), torch.min(wave))


def time_test(loader):
    import time

    MAX_ITER = 10

    # basic
    time_start = time.time()
    i = 0
    for waveforms, labels in loader:
        print(waveforms.shape)
        print(labels)
        i += 1
        if i >= MAX_ITER:
            break
    time_end = time.time()
    time_basic = time_end - time_start

    # cuda
    time_start = time.time()
    i = 0
    for waveforms, labels in loader:
        waveforms, labels = waveforms.cuda(), labels.cuda()
        print(waveforms.shape)
        print(labels)
        i += 1
        if i >= MAX_ITER:
            break
    time_end = time.time()
    time_cuda = time_end - time_start

    # non-blocking cuda
    time_start = time.time()
    i = 0
    for waveforms, labels in loader:
        waveforms, labels = waveforms.cuda(non_blocking=True), labels.cuda(non_blocking=True)
        print(waveforms.shape)
        print(labels)
        i += 1
        if i >= MAX_ITER:
            break
    time_end = time.time()
    time_cuda2 = time_end - time_start

    print()
    print(f"basic: {time_basic}")
    print(f"cuda: {time_cuda}")
    print(f"non-blocking cuda: {time_cuda2}")

if __name__ == "__main__":
    main()
