import logging
import os

import hydra
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig
from tokenizers import ByteLevelBPETokenizer, Tokenizer
from transformers import PreTrainedTokenizerFast

from src.processing import MessageProcessor

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
        extractor = DiffExtractor(**cfg.diff_extractor, data_format=cfg.data_format)
        extractor.extract_diffs(
            in_fname=os.path.join(cfg.paths.input_dir, "train_final"),
            out_fname=os.path.join(cfg.paths.tokenizer_dir, "diffs.txt"),
            n_examples=cfg.diff_extractor.n_train_examples if "n_train_examples" in cfg.diff_extractor else None,
            line_sep=cfg.line_sep,
        )

    # -----------------------------
    # -      train tokenizer      -
    # -----------------------------
    special_tokens = ["[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]"]
    additional_special_tokens = [cfg.line_sep, "\n"] + ["[LONG]"]
    if cfg.msg_tokens:
        additional_special_tokens += list(MessageProcessor.get_special_tokens().values())

    if cfg.tokenizer.configuration == "byte_level":
        tokenizer = ByteLevelBPETokenizer(**cfg.tokenizer.byte_level.tokenizer)
        tokenizer.train(
            **cfg.tokenizer.byte_level.train,
            files=[os.path.join(cfg.paths.tokenizer_dir, "diffs.txt")],
            special_tokens=special_tokens + additional_special_tokens
        )
    elif cfg.tokenizer.configuration == "custom":
        tokenizer = Tokenizer(instantiate(cfg.tokenizer.custom.tokenizer))
        if cfg.tokenizer.custom.normalizer:
            tokenizer.normalizer = instantiate(cfg.tokenizer.custom.normalizer)
        if cfg.tokenizer.custom.pre_tokenizer:
            tokenizer.pre_tokenizer = instantiate(cfg.tokenizer.custom.pre_tokenizer)
        if cfg.tokenizer.custom.decoder:
            tokenizer.decoder = instantiate(cfg.tokenizer.custom.decoder)
        trainer = instantiate(cfg.tokenizer.custom.trainer, special_tokens=special_tokens + additional_special_tokens)
        tokenizer.train(
            trainer=trainer,
            files=[os.path.join(cfg.paths.tokenizer_dir, "diffs.txt")],
        )
    else:
        raise ValueError("Unknown tokenizer configuration. Pass one of: `byte_level`, `custom`")

    logging.info("Saving tokenizer")

    os.makedirs(
        os.path.join(cfg.paths.tokenizer_dir, cfg.tokenizer.configuration, "transformers_format"), exist_ok=True
    )
    tokenizer.save(
        os.path.join(cfg.paths.tokenizer_dir, cfg.tokenizer.configuration, "diff_tokenizer.json"), pretty=True
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
        additional_special_tokens=additional_special_tokens,
    )
    transformers_tokenizer.save_pretrained(  # type: ignore[attr-defined]
        os.path.join(cfg.paths.tokenizer_dir, cfg.tokenizer.configuration, "transformers_format")
    )


if __name__ == "__main__":
    main()
