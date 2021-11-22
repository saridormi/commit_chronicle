from typing import Dict, Any
from pydriller import Modification, Commit


class CommitProcessor:
    @staticmethod
    def get_info_from_modification(m: Modification) -> Dict[str, str]:
        return {
            "change_type": str(m.change_type).split(".")[1],
            "old_path": m.old_path,
            "new_path": m.new_path,
            "diff": m.diff,
        }

    @staticmethod
    def process_commit(commit: Commit) -> Dict[str, Any]:
        """
        Return following information about commit:
        - author name & email
        - timestamp
        - hash
        - message
        - for each modified file:
            - which change was made (e.g. adding a new file, deleting a file, modifying an existing file)
            - old_path (relevant if file was deleted/renamed/copied)
            - new_path (relevant if file was added/renamed/copied)
            - diff
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
