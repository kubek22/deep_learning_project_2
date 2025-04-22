import os
import warnings
import random
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, WeightedRandomSampler


def create_sampler(train_dataset, alpha=1):
    # alpha parameter is responsible for balancing sample weight between classes
    # its value should be in [0, 1] interval
    # for alpha=0, all the samples have equal weight (no oversampling applied)
    # for alpha=1, all the classes are fully balanced

    train_labels = np.array([y for x, y in train_dataset.samples])
    unique, counts = np.unique(train_labels, return_counts=True)
    # weights = 1 / torch.tensor(counts, dtype=torch.float)
    # parametrized weights
    weights = 1 / torch.pow(torch.tensor(counts, dtype=torch.float), alpha)
    sample_weights = weights[train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
    return sampler

# deprecated
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
        warnings.warn(
            "AudioDataset is deprecated."
            "Please use SpectrogramDataset instead.",
            category=DeprecationWarning,
            stacklevel=2
        )

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


def get_paths(file):
    with open(file, 'r') as f:
        paths = [os.path.splitext(os.path.normpath(line.strip()))[0] + ".pt" for line in f.readlines()]
    return paths

class SpectrogramDataset(Dataset):
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
    MEAN = -10.408944129943848
    STD = 5.170785427093506

    val_paths = get_paths(VAL_LIST_PATH)
    test_paths = get_paths(TEST_LIST_PATH)

    def __init__(self, root_dir, transform=None, audio_length=AUDIO_LENGTH, set_type=TRAIN, augmentation=False, augmented_fraction=0.3, use_unknown=True):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.audio_length = audio_length
        self.set_type = set_type
        self.label_map = {}
        # keeping tensors in memory
        self.tensors = {}
        # noise tensors
        self.augmentation = augmentation
        self.augmented_fraction = augmented_fraction
        self.noise = []
        # determines using unknown classes
        self.use_unknown = use_unknown

        last_idx = 0
        for _, class_name in enumerate(sorted(os.listdir(root_dir))):
            idx = last_idx
            full_class_name = class_name # to capture real unknown class name
            class_path = os.path.join(root_dir, class_name)

            # loading noise tensors
            if class_name == self.NOISE_DIR:
                for file_name in os.listdir(class_path):
                    if file_name.endswith(".pt"):
                        file_path = os.path.join(class_path, file_name)
                        log_noise_spec = torch.load(file_path)
                        self.noise.append(log_noise_spec)
                continue

            if os.path.isdir(class_path):
                if class_name not in self.CLASSES:
                    # unknown class
                    if not self.use_unknown:
                        continue
                    class_name = self.UNKNOWN
                    # for unknown, get proper idx (one idx for all unknown)
                    idx = self.UNKNOWN_IDX
                else:
                    last_idx += 1
                if class_name not in self.label_map:
                    # condition in case of unknown classes
                    self.label_map[class_name] = idx
                for file_name in os.listdir(class_path):
                    # selecting only elements from given set
                    file_path = os.path.normpath(os.path.join(full_class_name, file_name))
                    file_in_val = file_path in self.val_paths
                    file_in_test = file_path in self.test_paths

                    if file_in_val and self.set_type != self.VAL:
                        continue
                    if file_in_test and self.set_type != self.TEST:
                        continue
                    if not (file_in_val or file_in_test) and self.set_type != self.TRAIN:
                        continue

                    if file_name.endswith(".pt"):
                        self.samples.append((os.path.join(class_path, file_name), idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        file_path, label = self.samples[idx]
        if idx not in self.tensors:
            log_spec = torch.load(file_path)
            log_spec_norm = (log_spec - SpectrogramDataset.MEAN) / SpectrogramDataset.STD
            self.tensors[idx] = log_spec_norm
        else:
            log_spec_norm = self.tensors[idx]

        if self.augmentation and random.random() < self.augmented_fraction:
            # transforming tensors to original spectrograms
            spec = torch.exp(log_spec_norm * SpectrogramDataset.STD + SpectrogramDataset.MEAN)
            log_noise = random.choice(self.noise)
            noise = torch.exp(log_noise)

            # cutting out noise segment
            start_idx = torch.randint(0, noise.size(2) - spec.size(2), (1,)).item()
            noise_segment = noise[:, :, start_idx:start_idx + spec.size(2)]

            # combining with noise
            noise_factor = random.uniform(0, 1)
            combined_spec = spec + noise_factor * noise_segment

            # transforming and standardizing
            log_combined_spec = torch.log(combined_spec + 1e-10)
            log_spec_norm = (log_combined_spec - SpectrogramDataset.MEAN) / SpectrogramDataset.STD

        return log_spec_norm, label

class BinaryDataset(SpectrogramDataset):
    SIGNAL_IDX = 0
    NO_SIGNAL_IDX = 1
    def __init__(self, root_dir, transform=None, audio_length=SpectrogramDataset.AUDIO_LENGTH, set_type=SpectrogramDataset.TRAIN, augmentation=False, augmented_fraction=0.3, use_unknown=True):
        super(BinaryDataset, self).__init__(root_dir, transform=transform, audio_length=audio_length, set_type=set_type, augmentation=augmentation, augmented_fraction=augmented_fraction, use_unknown=use_unknown)

        # updating signal indexes
        for i in range(len(self.samples)):
            file_path, idx = self.samples[i]
            if idx != self.UNKNOWN_IDX:
                idx = self.SIGNAL_IDX
                self.samples[i] = (file_path, idx)
        # updating noise indexes
        for i in range(len(self.samples)):
            file_path, idx = self.samples[i]
            if idx == self.UNKNOWN_IDX:
                idx = self.NO_SIGNAL_IDX
                self.samples[i] = (file_path, idx)

        # updating label mapping
        for class_name, idx in self.label_map.items():
            if idx != self.UNKNOWN_IDX:
                idx = self.SIGNAL_IDX
                self.label_map[class_name] = idx
        for class_name, idx in self.label_map.items():
            if idx == self.UNKNOWN_IDX:
                idx = self.NO_SIGNAL_IDX
                self.label_map[class_name] = idx
