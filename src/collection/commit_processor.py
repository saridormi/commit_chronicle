from typing import Dict, Sequence, Union

from pydriller import Commit, Modification


class CommitProcessor:
    @staticmethod
    def get_info_from_modification(m: Modification) -> Dict[str, str]:
        """
        Extracts specific information about single file modification.
        """
        return {
            "change_type": str(m.change_type).split(".")[1],
            "old_path": m.old_path,
            "new_path": m.new_path,
            "diff": m.diff,
        }

    @staticmethod
    def process_commit(commit: Commit) -> Dict[str, Union[Sequence[str], str]]:
        """
        Extracts specific information about commit.
        """
        res = {
            "author": (commit.author.name, commit.author.email),
            "date": commit.author_date.strftime("%d.%m.%Y %H:%M:%S"),
            "hash": commit.hash,
            "message": commit.msg,
            "mods": [],
        }

        for m in commit.modifications:
            res["mods"].append(CommitProcessor.get_info_from_modification(m))
        return res
