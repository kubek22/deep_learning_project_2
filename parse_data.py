import os
import torch
import torchaudio
import torchaudio.transforms as T


AUDIO_LENGTH = 16000
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def transform(waveform):
    waveform = waveform.to(device)
    # adapt to fixed size
    if waveform.size(1) < AUDIO_LENGTH:
        # padding with zeros at the end
        # waveform = torch.nn.functional.pad(waveform, (0, self.audio_length - waveform.size(1)))
        # or other possibility
        change_size = T.Resample(orig_freq=waveform.size(1), new_freq=AUDIO_LENGTH).to(device)
        waveform = change_size(waveform)

    # changing amplitudes into spectrogram
    spectrogram_transform = T.Spectrogram().to(device)
    spectrogram = spectrogram_transform(waveform)
    log_spec = torch.log(spectrogram + 1e-10)
    return log_spec.cpu()

def save_transformed_file(wav_file_path, class_path_dest, tensor):
    # file name without extension
    base_name = os.path.splitext(os.path.basename(wav_file_path))[0]
    file_path_dest = os.path.join(class_path_dest, f"{base_name}.pt")
    torch.save(tensor, file_path_dest)

def make_transformations(root_dir_source, root_dir_dest):
    # create root directory
    os.makedirs(root_dir_dest, exist_ok=True)
    print(f"Root directory {root_dir_dest} created\n")

    for _, class_name in enumerate(sorted(os.listdir(root_dir_source))):
        class_path = os.path.join(root_dir_source, class_name)
        print(f"Processing {class_path} ...")
        if os.path.isdir(class_path):
            # create a subdirectory
            class_path_dest = os.path.join(root_dir_dest, class_name)
            os.makedirs(class_path_dest, exist_ok=True)
            print(f"Directory {class_path_dest} created\n")

            for file_name in os.listdir(class_path):
                if file_name.endswith(".wav"):
                    file_path = os.path.join(class_path, file_name)
                    waveform, sample_rate = torchaudio.load(file_path)
                    output_tensor = transform(waveform)
                    save_transformed_file(file_path, class_path_dest, output_tensor)


def main():
    # run once to precompute the data and save them to files
    root_dir_source = "data/train/audio"
    root_dir_dest = "data/train/audio_transformed"
    make_transformations(root_dir_source, root_dir_dest)

if __name__ == "__main__":
    main()
