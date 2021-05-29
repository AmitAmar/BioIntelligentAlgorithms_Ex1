import sys
from utils import load_dataset, from_categorical

from ann import ANN


def main():
    if len(sys.argv) != 4:
        print(f"USAGE: {sys.argv[0]} <saved_model_path> <test_set_path> <predictions_output_path>")
        return

    saved_model_path = sys.argv[1]
    test_set_path = sys.argv[2]
    predictions_output_path = sys.argv[3]

    ann = ANN.load(saved_model_path)
    test_data, _ = load_dataset(test_set_path)
    print(f"{test_data[0].shape}")

    predictions_vectors = [ann.predict(test_data[i]) for i in range(len(test_data))]
    # We need to add one to the prediction, because the provided datasets' tags are 1-based
    predictions = [from_categorical(predictions_vectors[i]) + 1 for i in range(len(predictions_vectors))]
    print(predictions)

    with open(predictions_output_path, "w") as file_:
        file_.write("\n".join([str(x) for x in predictions]))


if __name__ == "__main__":
    main()
