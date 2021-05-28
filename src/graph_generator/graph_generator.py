from os import listdir
from os.path import isfile, join

import matplotlib.pyplot as plt
from ann_sample import AnnSample

PATH = r'../../models/'


def get_all_files_names_in_folder(path):
    return [f for f in listdir(path) if isfile(join(path, f))]


def parse_samples(files_names):
    ann_samples = list()
    for name in files_names:
        chunks = name.split("_")
        ann_samples.append(AnnSample(chunks[0], chunks[1], chunks[2]))
    ann_samples.sort(key=lambda sample: sample.index)

    return ann_samples


def main():
    files_names = get_all_files_names_in_folder(PATH)

    samples = parse_samples(files_names)

    x = [sample.train for sample in samples]
    y = [sample.validate for sample in samples]

    plt.plot(x, y)
    plt.xlabel("TRAIN")
    plt.ylabel("VALIDATE")
    plt.title("Train VS Validate")
    plt.show()


if __name__ == '__main__':
    main()
