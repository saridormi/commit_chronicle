import os
import pandas as pd
import logging
import argparse
from post_deduplication_utils import PostDeduplicationProcessor, MessageFilter, DiffFilter

logging.basicConfig(
    filename="../../logs/prepare_for_tokenization.log",
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script processes clones output from SourcererCC",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw_data_dir", type=str, default="../extracted_data/",
                        help="path to directory with commits data")
    parser.add_argument("--deduplication_dir", type=str, default="../deduplication/",
                        help="path to directory with clones results")
    parser.add_argument("--chunksize", type=int, default=5000, help="# of examples to process at one step")
    args = parser.parse_args()
    # --------------------------------------
    #             drop duplicates
    # --------------------------------------
    if "train_no_duplicates.csv" not in os.listdir(args.raw_data_dir):
        dp = PostDeduplicationProcessor(raw_data_dir=args.raw_data_dir,
                                        deduplication_dir=args.deduplication_dir)
        dp(msg_filename="messages_clones.csv",
           diff_filename="diffs_clones.csv",
           raw_msg_filename="results_messages_100_multi.pairs",
           raw_diff_filename="results_diffs_100_multi.pairs")
    # --------------------------------------
    #             filter messages
    # --------------------------------------
    #if f"filtered_train.csv" not in os.listdir(args.raw_data_dir):
    logging.info(f"Filtering messages from train")
    MessageFilter.filter(input_filename=os.path.join(args.raw_data_dir, "train_no_duplicates.csv"),
                         output_filename=os.path.join(args.raw_data_dir, "filtered_train.csv"),
                         chunksize=args.chunksize)
    for part in ["val", "test", "val_original", "test_original"]:
        #if f"filtered_{part}.csv" not in os.listdir(args.raw_data_dir):
        logging.info(f"Filtering messages from {part}")
        MessageFilter.filter(input_filename=os.path.join(args.raw_data_dir, f"{part}_no_outliers.csv"),
                             output_filename=os.path.join(args.raw_data_dir, f"filtered_{part}.csv"),
                             chunksize=args.chunksize)
    # --------------------------------------
    #               drop NaNs
    # --------------------------------------
    for part in ["train", "val", "test", "val_original", "test_original"]:
        df = pd.read_csv(os.path.join(args.raw_data_dir, f"filtered_{part}.csv"))
        df = df.dropna()
        df[["id", "author", "date", "hash", "message", "diff", "repo"]].to_csv(
            os.path.join(args.raw_data_dir, f"{part}_final.csv"), index=False, header=True)
    # --------------------------------------
    #             filter diffs
    # --------------------------------------
    for part in ["train", "val", "test", "val_original", "test_original"]:
        if f"{part}_final.csv" not in os.listdir(args.raw_data_dir):
            logging.info(f"Filtering diffs from {part}")
            DiffFilter.filter(input_filename=os.path.join(args.raw_data_dir, f"filtered_{part}.csv"),
                              output_filename=os.path.join(args.raw_data_dir, f"{part}_final.csv"),
                              chunksize=args.chunksize)

    # --------------------------------------
    #               drop NaNs
    # --------------------------------------
    for part in ["train", "val", "test", "val_original", "test_original"]:
        df = pd.read_csv(os.path.join(args.raw_data_dir, f"{part}_final.csv"))
        df = df.dropna()
        df[["id", "author", "date", "hash", "message", "diff", "repo"]].to_csv(
        os.path.join(args.raw_data_dir, f"{part}_final.csv"), index=False, header=True)
