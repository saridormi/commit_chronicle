import logging
import os

import hydra
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Sequence, WhitespaceSplit
from tokenizers.processors import TemplateProcessing

from .tokenization import DiffExtractor


@hydra.main(config_path="../configs", config_name="train_tokenizer")
def main(cfg: DictConfig) -> None:
    for key in cfg.paths:
        cfg.paths[key] = to_absolute_path(cfg.paths[key])
        os.makedirs(cfg.paths[key], exist_ok=True)

    logging.info("======= Using config =======")
    logging.info(cfg)

    # -----------------------------
    # -  prepare training data    -
    # -----------------------------

    n_examples = cfg.n_train_examples if "n_train_examples" in cfg else None
    extractor = DiffExtractor(**cfg.diff_extractor, data_format=cfg.data_format)
    extractor.extract_diffs(
        in_fname=os.path.join(cfg.paths.input_dir, "tokenization", "train_final"),
        out_fname=os.path.join(cfg.paths.tokenizer_dir, "diffs.txt"),
        n_examples=n_examples,
    )

    # -----------------------------
    # -      train tokenizer      -
    # -----------------------------

    tokenizer = Tokenizer(instantiate(cfg.tokenizer))
    tokenizer.pre_tokenizer = Sequence([instantiate(cfg.pre_tokenizer), WhitespaceSplit()])
    tokenizer.post_processor = TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[
            ("[CLS]", 1),
            ("[SEP]", 2),
        ],
    )

    trainer = instantiate(cfg.trainer)
    tokenizer.train(trainer, [os.path.join(cfg.paths.tokenizer_dir, "diffs.txt")])

    logging.info("Saving tokenizer")
    tokenizer.save(os.path.join(cfg.paths.tokenizer_dir, "diff_tokenizer.json"), pretty=True)


if __name__ == "__main__":
    main()
