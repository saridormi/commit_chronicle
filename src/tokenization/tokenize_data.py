import hydra
import os
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from src.tokenization.utils import TrainingProcessor


@hydra.main(config_path="configs", config_name="tokenization_config")
def main(cfg: DictConfig) -> None:
    cfg.training_processor.diff_tokenizer_name_or_path = to_absolute_path(
        cfg.training_processor.diff_tokenizer_name_or_path
    )
    for path in cfg.paths:
        cfg.paths[path] = to_absolute_path(cfg.paths[path])
    print(OmegaConf.to_yaml(cfg))
    processor = TrainingProcessor(**cfg.training_processor)

    for part in ["train", "val", "test", "val_original", "test_original"]:
        processor(
            input_filename=os.path.join(cfg.paths.input_root_dir, f"{part}_final.csv"),
            output_dir=cfg.paths.output_root_dir,
            part=part,
        )


if __name__ == "__main__":
    main()
