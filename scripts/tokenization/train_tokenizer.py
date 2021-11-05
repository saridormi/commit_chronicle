import hydra
import os
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tokenizers import Tokenizer


@hydra.main(config_path="configs", config_name="train_tokenizer_config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    tokenizer = Tokenizer(instantiate(cfg.tokenizer))
    tokenizer.pre_tokenizer = instantiate(cfg.pre_tokenizer)

    trainer = instantiate(cfg.trainer)
    files = [
        to_absolute_path(os.path.join(cfg.paths.data_root_dir, f"{part}.txt"))
        for part in ["train", "val", "test", "val_original", "test_original"]
    ]
    tokenizer.train(trainer, files)
    tokenizer.save(to_absolute_path(cfg.paths.tokenizer_root_dir))


if __name__ == "__main__":
    main()
