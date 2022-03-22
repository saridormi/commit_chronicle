import hydra
import os
import logging
from hydra.utils import instantiate, to_absolute_path
from omegaconf import DictConfig
from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Sequence, WhitespaceSplit
from tokenizers.processors import TemplateProcessing
from src.tokenization.train_tokenizer_utils import Lexer


@hydra.main(config_path=".", config_name="train_tokenizer_config")
def main(cfg: DictConfig) -> None:
    for key in cfg.paths:
        cfg.paths[key] = to_absolute_path(cfg.paths[key])
        os.makedirs(cfg.paths[key], exist_ok=True)

    parts = ["train"] + sorted(
        [
            part.split(".")[0]
            for part in os.listdir(cfg.paths.input_dir)
            if not os.path.isdir(os.path.join(cfg.paths.input_dir, part)) and "train" not in part
        ]
    )

    logging.info("======= Using config =======")
    logging.info(cfg)

    os.makedirs(os.path.join(cfg.paths.input_dir, "lexed"), exist_ok=True)
    os.makedirs(os.path.join(cfg.paths.input_dir, "lexed_diffs_only"), exist_ok=True)

    # -----------------------------
    # -         lex diffs         -
    # -----------------------------

    if "fnames" in cfg:
        fnames = [to_absolute_path(fname) for fname in cfg.fnames]
    else:
        fnames = []

        lexer = Lexer(**cfg.lexer, data_format=cfg.data_format, logger_name="lexer")
        for part in parts:
            part_fname = os.path.join(cfg.paths.input_dir, "lexed_diffs_only", f"{part}.txt")
            os.makedirs(os.path.join(cfg.paths.percentile_dir, part), exist_ok=True)

            percentile_dir = None
            if part != "train":
                percentile_dir = os.path.join(cfg.paths.percentile_dir, "train")

            logging.info(f"Pretokenizing {part}")
            lexer(
                in_fname=os.path.join(cfg.paths.input_dir, "filtered_diffs", part),
                out_fname=os.path.join(cfg.paths.input_dir, "lexed", part),
                diffs_out_fname=part_fname,
                prepare_literals_len_dir=os.path.join(cfg.paths.percentile_dir, part),
                prepare_percentile_dir=percentile_dir,
            )

            fnames.append(part_fname)

    # --------------------------------
    # -        train tokenizer       -
    # --------------------------------

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
    tokenizer.train(trainer, fnames)

    logging.info("Saving tokenizer")
    tokenizer.save(os.path.join(cfg.paths.tokenizer_dir, "diff_tokenizer.json"), pretty=True)


if __name__ == "__main__":
    main()
