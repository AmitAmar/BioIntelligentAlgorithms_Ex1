from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
from ann_sample import AnnSample

MODELS_PATH = r'../../models/'


def get_all_files_names_in_folder(path):
    return [f.replace('.ann', '') for f in listdir(path) if isfile(join(path, f))]


def parse_samples(files_names):
    ann_samples = list()
    for name in files_names:
        chunks = name.split("_")
        ann_samples.append(AnnSample(chunks[0], chunks[1], chunks[2]))
    ann_samples.sort(key=lambda sample: sample.index)

    return ann_samples


def draw_graph(samples):
    epochs = [sample.index for sample in samples]
    train = [float(sample.train) for sample in samples]
    validate = [float(sample.validate) for sample in samples]

    plt.plot(epochs, train, label='train')
    plt.plot(epochs, validate, label='validate')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train & Validate")
    plt.legend()
    plt.show()


def main():
    files_names = get_all_files_names_in_folder(MODELS_PATH)
    samples = parse_samples(files_names)
    draw_graph(samples)


if __name__ == '__main__':
    main()
