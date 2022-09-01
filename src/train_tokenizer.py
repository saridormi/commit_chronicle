import logging
import os

import hydra
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig
from tokenizers import ByteLevelBPETokenizer, Tokenizer
from tokenizers.pre_tokenizers import Sequence, WhitespaceSplit
from transformers import PreTrainedTokenizerFast

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
    if "extract" not in cfg or cfg.extract:
        n_examples = cfg.n_train_examples if "n_train_examples" in cfg else None
        extractor = DiffExtractor(**cfg.diff_extractor, data_format=cfg.data_format)
        extractor.extract_diffs(
            in_fname=os.path.join(cfg.paths.input_dir, "tokenization", "train_final"),
            out_fname=os.path.join(cfg.paths.tokenizer_dir, "diffs.txt"),
            n_examples=n_examples,
            line_sep=cfg.line_sep,
        )

    # -----------------------------
    # -      train tokenizer      -
    # -----------------------------
    special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]", "[LONG]", "\n"] + [cfg.line_sep]
    if cfg.tokenizer_configuration == "byte_level":
        tokenizer = ByteLevelBPETokenizer(**cfg.byte_level.tokenizer)
        tokenizer.train(
            **cfg.byte_level.train,
            files=[os.path.join(cfg.paths.tokenizer_dir, "diffs.txt")],
            special_tokens=special_tokens
        )
    elif cfg.tokenizer_configuration == "whitespace_byte_level":
        tokenizer = ByteLevelBPETokenizer(**cfg.byte_level.tokenizer)
        tokenizer.pre_tokenizer = Sequence([WhitespaceSplit(), tokenizer.pre_tokenizer])
        tokenizer.train(
            **cfg.byte_level.train,
            files=[os.path.join(cfg.paths.tokenizer_dir, "diffs.txt")],
            special_tokens=special_tokens
        )
    elif cfg.tokenizer_configuration == "custom":
        tokenizer = Tokenizer(instantiate(cfg.custom.tokenizer))
        if cfg.custom.normalizer:
            tokenizer.normalizer = instantiate(cfg.custom.normalizer)
        if cfg.custom.pre_tokenizer:
            tokenizer.pre_tokenizer = instantiate(cfg.custom.pre_tokenizer)
        if cfg.custom.decoder:
            tokenizer.decoder = instantiate(cfg.custom.decoder)
        trainer = instantiate(cfg.custom.trainer, special_tokens=special_tokens)
        tokenizer.train(
            trainer=trainer,
            files=[os.path.join(cfg.paths.tokenizer_dir, "diffs.txt")],
        )
    else:
        raise ValueError("Unknown tokenizer configuration")

    logging.info("Saving tokenizer")

    os.makedirs(
        os.path.join(cfg.paths.tokenizer_dir, cfg.tokenizer_configuration, "transformers_format"), exist_ok=True
    )
    tokenizer.save(
        os.path.join(cfg.paths.tokenizer_dir, cfg.tokenizer_configuration, "diff_tokenizer.json"), pretty=True
    )
    transformers_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        bos_token="[BOS]",
        pad_token="[PAD]",
        eos_token="[EOS]",
        unk_token="[UNK]",
        cls_token="[BOS]",
        sep_token="[EOS]",
        mask_token="[MASK]",
        additional_special_tokens=["[LONG]", "\n"] + [cfg.line_sep],
    )
    transformers_tokenizer.save_pretrained(
        os.path.join(cfg.paths.tokenizer_dir, cfg.tokenizer_configuration, "transformers_format")
    )


if __name__ == "__main__":
    main()
