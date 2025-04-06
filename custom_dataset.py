import os
import torch
import numpy as np
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class AudioDataset(Dataset):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"
    CLASSES = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
    UNKNOWN = "unknown"
    UNKNOWN_IDX = 10
    AUDIO_LENGTH = 16000
    NOISE_DIR = "_background_noise_"
    VAL_LIST_PATH = "data/train/validation_list.txt"
    TEST_LIST_PATH = "data/train/testing_list.txt"

    with open(VAL_LIST_PATH, 'r') as f:
        val_paths = [os.path.normpath(line.strip()) for line in f.readlines()]
    val_paths = set(val_paths)

    with open(TEST_LIST_PATH, 'r') as f:
        test_paths = [os.path.normpath(line.strip()) for line in f.readlines()]
    test_paths = set(test_paths)

    def __init__(self, root_dir, transform=None, audio_length=AUDIO_LENGTH, set_type=TRAIN):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.audio_length = audio_length
        self.set_type = set_type
        self.label_map = {}
        last_idx = 0
        for _, class_name in enumerate(sorted(os.listdir(root_dir))):
            idx = last_idx
            full_class_name = class_name # to capture real unknown class name
            if class_name == AudioDataset.NOISE_DIR:
                # ignoring noise directory
                continue
            class_path = os.path.join(root_dir, class_name)
            if os.path.isdir(class_path):
                if class_name not in AudioDataset.CLASSES:
                    # unknown class
                    class_name = AudioDataset.UNKNOWN
                    # for unknown, get proper idx (one idx for all unknown)
                    idx = AudioDataset.UNKNOWN_IDX
                else:
                    last_idx += 1
                if class_name not in self.label_map:
                    # condition in case of unknown classes
                    self.label_map[class_name] = idx
                for file_name in os.listdir(class_path):
                    # selecting only elements from given set
                    file_path = os.path.normpath(os.path.join(full_class_name, file_name))
                    file_in_val = file_path in AudioDataset.val_paths
                    file_in_test = file_path in AudioDataset.test_paths

                    if file_in_val and self.set_type != AudioDataset.VAL:
                        continue
                    if file_in_test and self.set_type != AudioDataset.TEST:
                        continue
                    if not (file_in_val or file_in_test) and self.set_type != AudioDataset.TRAIN:
                        continue

                    if file_name.endswith(".wav"):
                        self.samples.append((os.path.join(class_path, file_name), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        waveform, sample_rate = torchaudio.load(file_path)

        # adapt to fixed size
        if waveform.size(1) < self.audio_length:
            # padding with zeros at the end
            # waveform = torch.nn.functional.pad(waveform, (0, self.audio_length - waveform.size(1)))
            # or other possibility
            change_size = T.Resample(orig_freq=waveform.size(1), new_freq=self.audio_length)
            waveform = change_size(waveform)

        # changing amplitudes into spectrogram
        spectrogram = T.Spectrogram()(waveform)
        log_spec = torch.log(spectrogram + 1e-10)

        # Apply transformation if available
        if self.transform:
            log_spec = self.transform(log_spec)

        return log_spec, label


def main():
    # main function is required for multiprocessing
    compute_train_stats()

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


def compute_train_stats():
    print("starting processing...")
    data_path = "data/train/audio"
    dataset = AudioDataset(data_path, set_type=AudioDataset.TRAIN)
    print("creating dataloader...")
    train_loader = DataLoader(
        dataset,
        batch_size=512,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sum = torch.tensor(0.0, device=device)
    num_pixels = torch.tensor(0.0, device=device)

    for log_spec, _ in train_loader:
        print("new batch processing...")
        log_spec = log_spec.to(device)
        log_spec = log_spec.view(log_spec.size(0), -1)

        sum += log_spec.sum()
        num_pixels += log_spec.numel()

    mean = sum / num_pixels

    print(mean)
    # print(std)



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
