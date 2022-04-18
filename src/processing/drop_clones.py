import logging
import os

import hydra
from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from src.processing.utils import FinalProcessor, PostDeduplicationProcessor


@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig) -> None:
    for key in cfg.paths:
        cfg.paths[key] = to_absolute_path(cfg.paths[key])
        os.makedirs(cfg.paths[key], exist_ok=True)

    logging.info("======= Using config =======")
    logging.info(cfg)

    parts = ["train"] + sorted(
        [
            part.split(".")[0]
            for part in os.listdir(cfg.paths.input_dir)
            if not os.path.isdir(os.path.join(cfg.paths.input_dir, part)) and "train" not in part
        ]
    )

    print(parts)

    # ----------------------------
    # -       drop clones        -
    # ----------------------------

    processor = PostDeduplicationProcessor(
        **cfg.post_deduplication_processor, data_format=cfg.data_format, logger_name="postdedupl_processor"
    )
    processor(
        in_fname=os.path.join(cfg.paths.input_dir, "lexed", "train"),
        out_fname=os.path.join(cfg.paths.input_dir, "lexed", "train_no_duplicates"),
        prepare_in_path=os.path.join(cfg.paths.input_dir, "lexed"),
        prepare_diff_clones_fname="results_messages_100_multi.pairs",
        prepare_msg_clones_fname="results_diffs_100_multi.pairs",
        prepare_deduplication_dir=cfg.paths.deduplication_dir,
        prepare_parts=parts,
    )
    processor(
        in_fname=os.path.join(cfg.paths.input_dir, "tokenization", "train"),
        out_fname=os.path.join(cfg.paths.input_dir, "tokenization", "train_no_duplicates"),
        prepare_is_ready=True,
        prepare_in_path=os.path.join(cfg.paths.input_dir, "lexed"),
        prepare_diff_clones_fname="results_messages_100_multi.pairs",
        prepare_msg_clones_fname="results_diffs_100_multi.pairs",
        prepare_deduplication_dir=cfg.paths.deduplication_dir,
        prepare_parts=parts,
    )

    # ---------------------------------
    # -       work on metadata        -
    # ---------------------------------

    processor = FinalProcessor(**cfg.final_processor, data_format=cfg.data_format, logger_name="final_processor")
    for part in parts:
        logging.info(f"Converting authors in {part}")

        processor(
            in_fname=os.path.join(cfg.paths.input_dir, "lexed", part + ("_no_duplicates" if part == "train" else "")),
            out_fname=os.path.join(cfg.paths.input_dir, f"{part}_final"),
            prepare_license_in_fname=os.path.join(cfg.paths.licenses_dir, "repo_license_map.json"),
            prepare_in_fnames=[os.path.join(cfg.paths.input_dir, part) for part in parts],
        )

        processor(
            in_fname=os.path.join(
                cfg.paths.input_dir, "tokenization", part + ("_no_duplicates" if part == "train" else "")
            ),
            out_fname=os.path.join(cfg.paths.input_dir, "tokenization", f"{part}_final"),
            prepare_license_in_fname=os.path.join(cfg.paths.licenses_dir, "repo_license_map.json"),
            prepare_in_fnames=[os.path.join(cfg.paths.input_dir, part) for part in parts],
        )


if __name__ == "__main__":
    main()
