from typing import Dict, Union, List, Optional
from pydriller import ModificationType, Modification, Commit


class CommitProcessor:
    @staticmethod
    def preprocess_type(m: Modification) -> str:
        """
        Returns a short summary based on ModificationType to make data format closer to git diff --patch
        (which is the default).
        """
        if m.change_type == ModificationType.ADD:
            return f"""new file {m.new_path}\n"""

        if m.change_type == ModificationType.DELETE:
            return f"""deleted file {m.old_path}\n"""

        if m.change_type == ModificationType.RENAME:
            return f"""rename from {m.old_path}\nrename to {m.new_path}\n"""

        if m.change_type == ModificationType.COPY:
            return f"""copy from {m.old_path}\ncopy to {m.new_path}\n"""

        if m.change_type == ModificationType.MODIFY:
            return f"""{m.new_path}\n"""

    @staticmethod
    def get_diff_from_modification(m: Modification) -> Optional[str]:
        """
        1) Generates prefix based on ModificationType
        2) Concatenates it with diff
        """
        if m.change_type == ModificationType.UNKNOWN:
            return None

        prefix = CommitProcessor.preprocess_type(m)

        return prefix + m.diff

    @staticmethod
    def process_commit(commit: Commit) -> Dict[str, Union[str, List[str]]]:
        res = {
            "author": f"{commit.author.name}[SEP]{commit.author.email}",
            "date": commit.author_date.strftime("%d.%m.%Y %H:%M:%S"),
            "hash": commit.hash,
            "message": commit.msg,
            "diff": [],
        }

        for m in commit.modifications:
            diff = CommitProcessor.get_diff_from_modification(m)
            if diff is not None:
                res["diff"].append(diff)
        res["diff"] = " ".join(res["diff"])
        return res
