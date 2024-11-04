import argparse

from pipeline import initialize

parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--raw_data_folder", default="", type=str)
parser.add_argument("--api_key", default="", type=str)

args = parser.parse_args()

if __name__ == "__main__":
    initialize(api_key=args.api_key, folder_path=args.raw_data_folder)