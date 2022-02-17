import hydra
import os
import logging
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from src.tokenization.data_tokenization_utils import TrainingProcessor


@hydra.main(config_path=".", config_name="data_tokenization_config")
def main(cfg: DictConfig) -> None:
    for key in cfg.paths:
        cfg.paths[key] = to_absolute_path(cfg.paths[key])

    logging.info("======= Using config =======")
    logging.info(cfg)

    processor = TrainingProcessor(**cfg.training_processor, diff_tokenizer_path=cfg.paths.diff_tokenizer_path)
    for part in ["train", "val", "test", "val_original", "test_original"]:
        processor(
            in_fname=os.path.join(cfg.paths.input_dir, "lexed", f"{part}.jsonl"),
            output_dir=cfg.paths.output_dir,
            part=part,
        )


if __name__ == "__main__":
    main()
