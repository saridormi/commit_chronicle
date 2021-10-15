import os
import logging
import argparse
from pre_deduplication_utils import DeduplicationProcessor, OutliersProcessor, unite_to_one_file

logging.basicConfig(
    filename="../logs/prepare_for_deduplication.log",
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="This script does all necessary preprocessing before deduplication via SourcererCC",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input_dir", type=str, help="path to directory with collected commit data")
    parser.add_argument("--percentile_dir", type=str, help="path directory to save percentiles & numbers of tokens")
    parser.add_argument("--deduplication_dir", type=str, help="path to save processed files")
    parser.add_argument("--chunksize", type=int, default=1000, help="# of examples to process at one step")
    args = parser.parse_args()
    os.makedirs(args.input_dir, exist_ok=True)
    os.makedirs(args.percentile_dir, exist_ok=True)
    os.makedirs(args.deduplication_dir, exist_ok=True)

    # --------------------------------------
    # unite collected data into single file
    # --------------------------------------
    for part in ["train", "val", "test", "val_original", "test_original"]:
        if f"{part}.csv" not in os.listdir(args.input_dir):
            logging.info(f"Processing raw {part} data")
            unite_to_one_file(
                input_dir=os.path.join(args.input_dir, part),
                output_filename=os.path.join(args.input_dir, f"{part}.csv"),
                chunksize=args.chunksize,
            )

    # -----------------------------------------------------------------------------------
    # calculate percentiles of # tokens and drop outliers <= or >= specified percentiles
    # -----------------------------------------------------------------------------------
    outliers_processor = OutliersProcessor(lower_percentile=0.01, upper_percentile=0.95)
    if "train_no_outliers.csv" not in os.listdir(args.input_dir):
        logging.info("Dropping outliers from train")
        outliers_processor(
            input_filename=os.path.join(args.input_dir, "train.csv"),
            n_tokens_dir=os.path.join(args.percentile_dir, "train"),
            output_filename=os.path.join(args.input_dir, "train_no_outliers.csv"),
            chunksize=args.chunksize,
        )
    for part in ["val", "test", "val_original", "test_original"]:
        if f"{part}_no_outliers.csv" not in os.listdir(args.input_dir):
            logging.info(f"Dropping outliers from {part}")
            outliers_processor(
                input_filename=os.path.join(args.input_dir, f"{part}.csv"),
                n_tokens_dir=os.path.join(args.percentile_dir, part),
                # use percentiles calculated on train
                percentile_dir=os.path.join(args.percentile_dir, "train"),
                output_filename=os.path.join(args.input_dir, f"{part}_no_outliers.csv"),
                chunksize=args.chunksize,
            )

    # ------------------------------------------------
    # preprocess data into format SourcererCC expects
    # ------------------------------------------------
    for part_id, part in [(1, "train"), (2, "val"), (3, "test"), (4, "val_original"), (5, "test_original")]:
        dp = DeduplicationProcessor(project_id=part_id)
        if f"{part}_message.txt" not in os.listdir(args.deduplication_dir):
            logging.info(f"Processing messages from {part} into SourcererCC format")
            dp.preprocess(
                input_filename=os.path.join(args.input_dir, f"{part}_no_outliers.csv"),
                output_filename=os.path.join(args.deduplication_dir, f"{part}_message.txt"),
                chunksize=args.chunksize,
                diff_mode=False,
            )
            if f"{part}_diff.txt" not in os.listdir(args.deduplication_dir):
                logging.info(f"Processing diffs from {part} into SourcererCC format")
                dp.preprocess(
                    input_filename=os.path.join(args.input_dir, f"{part}_no_outliers.csv"),
                    output_filename=os.path.join(args.deduplication_dir, f"{part}_diff.txt"),
                    chunksize=args.chunksize,
                    diff_mode=True,
                )

    with open(os.path.join(args.deduplication_dir, "res_diff.txt"), "w") as outfile:
        for part in ["train", "val", "test", "val_original", "test_original"]:
            with open(os.path.join(args.deduplication_dir, f"{part}_diff.txt"), "r") as infile:
                for line in infile:
                    outfile.write(line)

    with open(os.path.join(args.deduplication_dir, "res_message.txt"), "w") as outfile:
        for part in ["train", "val", "test", "val_original", "test_original"]:
            with open(os.path.join(args.deduplication_dir, f"{part}_message.txt"), "r") as infile:
                for line in infile:
                    outfile.write(line)
