from os import listdir
from os.path import isfile, join
import numpy as np
import sys

import ipdb
import matplotlib.pyplot as plt
from ann_sample import AnnSample


def get_all_files_names_in_folder(path):
    return [f.replace('.ann', '') for f in listdir(path) if isfile(join(path, f))]


def parse_samples(files_names):
    ann_samples = list()
    for name in files_names:
        chunks = name.split("_")
        try:
            ann_samples.append(AnnSample(chunks[0], float(chunks[1]), float(chunks[2])))
        except IndexError:
            continue
        except ValueError:
            continue
    ann_samples.sort(key=lambda sample: int(sample.index))

    return ann_samples


def draw_graph(samples):
    epochs = [int(sample.index) for sample in samples]
    train = [sample.train for sample in samples]
    validate = [sample.validate for sample in samples]

    plt.plot(epochs, train, label='train')
    plt.plot(epochs, validate, label='validate')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Train & Validate")
    plt.xticks(np.arange(min(epochs), max(epochs)+1, 10.0))
    plt.legend()
    plt.show()


def main():
    if not len(sys.argv) == 2:
        print(f"USAGE: {sys.argv[0]} <models_dir>")
        return

    models_dir = sys.argv[1]

    files_names = get_all_files_names_in_folder(models_dir)
    samples = parse_samples(files_names)
    draw_graph(samples)


if __name__ == '__main__':
    main()
