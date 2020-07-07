import argparse


def preprocess():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True)
    parser.add_argument("--output_dir", default=None, type=str, required=True)
    parser.add_argument("--feature_types", default=None, type=str, required=True)
    args = parser.parse_args()


if __name__ == "__main__":
    preprocess()