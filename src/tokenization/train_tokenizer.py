import hydra
import os
import logging
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tokenizers import Tokenizer
from src.tokenization.utils import Lexer


@hydra.main(config_path="configs", config_name="train_tokenizer_config")
def main(cfg: DictConfig) -> None:
    logging.info("Tokenizer config")
    logging.info(OmegaConf.to_yaml(cfg))
    tokenizer = Tokenizer(instantiate(cfg.tokenizer))

    lexer = Lexer(sep_token=cfg.pre_tokenizer.pattern)
    fnames = []
    for part in ["train", "val", "test", "val_original", "test_original"]:
        part_fname = to_absolute_path(os.path.join(cfg.paths.data_dir, f"diffs/{part}.txt"))
        if not os.path.exists(part_fname):
            logging.info(f"Pretokenizing {part}")
            lexer(
                input_filename=to_absolute_path(os.path.join(cfg.paths.data_dir, f"{part}_final.csv")),
                output_filename=to_absolute_path(os.path.join(cfg.paths.data_dir, f"{part}_final_pretokenized.csv")),
                save_diffs=True,
                diff_filename=to_absolute_path(os.path.join(cfg.paths.data_dir, f"diffs/{part}.txt")),
                chunksize=cfg.chunksize,
            )
        fnames.append(part_fname)

    tokenizer.pre_tokenizer = instantiate(cfg.pre_tokenizer)

    trainer = instantiate(cfg.trainer)
    tokenizer.train(fnames, trainer)
    tokenizer.save(to_absolute_path(cfg.paths.tokenizer_fname))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.FileHandler("tokenizer_training.log"), logging.StreamHandler()],
    )

    main()
