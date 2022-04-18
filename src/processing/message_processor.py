import re
from string import punctuation

import pandas as pd
from joblib import Parallel, delayed

from ..utils import BaseProcessor


class MessageProcessor(BaseProcessor):
    """
    This class is used to delete undesirable patterns from messages and filter messages.

    * Reused regexes for deleting emails, urls and SHA from
    Liu, Zhongxin, et al. "Automatic generation of pull request descriptions."
    2019 34th IEEE/ACM International Conference on Automated Software Engineering (ASE). IEEE, 2019.

    * Reused regexes for filtering bot and trivial messages from
    Liu, Zhongxin, et al. "Neural-machine-translation-based commit message generation: how far are we?."
    Proceedings of the 33rd ACM/IEEE International Conference on Automated Software Engineering. 2018.
    """

    @staticmethod
    def _filter_emails(message: str) -> str:
        return re.sub(r"(^|\s)<[\w.-]+@(?=[a-z\d][^.]*\.)[a-z\d.-]*[^.]>", "", message)

    @staticmethod
    def _filter_urls(message: str) -> str:
        return re.sub(r"https?://[-a-zA-Z0-9@:%._+~#?=/]+(?=($|[^-a-zA-Z0-9@:%._+~#?=/]))", "", message)

    @staticmethod
    def _filter_at_pattern(message: str) -> str:
        return re.sub(r"@\S+", "", message)

    @staticmethod
    def _filter_sha(message: str) -> str:
        x = re.sub(r"(^|\s)[\dA-Fa-f-]{7,}(?=(\s|$))", "", message)
        x = re.sub(r"(ref:)[\dA-Fa-f-]{7,}(?=(\s|$))", "", x)  # from yandex repos
        x = re.sub(r"\bI[0-9a-fA-F]{6,40}\b", "", x)  # gerrit
        return x

    @staticmethod
    def _filter_issue_ref(message: str) -> str:
        """
        Deletes issue numbers from the following patterns:

        * #123, [#123], (#123), <#123>
        * GH-123
        * gh-123
        * [#123]
        * CAT-123 (Jira project id)
        """
        x = re.sub("[\[\(<]?#[\d]+[\]\)>]?", "", message)
        x = re.sub("GH-[\d]+", "", x)
        x = re.sub("gh-[\d]+", "", x)
        x = re.sub("([A-Z][A-Z0-9]+-[0-9]+)", "", x)
        return x

    @staticmethod
    def _filter_signature(message: str) -> str:
        """
        Filters various signatures from messages

        * Not sure about specific tools/repos, but these kinds of signatures appear quite often
            * `Signed-off-by: <username>`
            * `Co-authored-by: <username>`
            * `Also-by: <username>`
            * `Reviewed-by: <username>`
            * `Former commit id: <id>`
        * https://github.com/google/moe: `Created by MOE: <some link>\nMOE_MIGRATED_REVID=<some number>`
        * https://github.com/facebook/fbshipit:
            * `Differential Revision: <some number>`
            * `Pulled By: <username>`
            * `fbshipit-source-id: <some sha-like string>`
        * https://github.com/google/copybara:
            * `BUG=<some number>`
            * `FIXES=<some number>`
            * `Change-Id: <some sha-like string>`
            * `PiperOrigin-RevId: <some number>`
            * `BAZEL_VERSION_REV_ID: <some number>`
        """
        x = re.sub(
            r"(signed(-| |)off(-| |)by|co(-| |)authored(-| |)by|also(-| |)by|reviewed(-| |)by|pulled(-| |)by|former("
            r"-| |)commit(-| |)id).*?(\n|$)",
            "",
            message,
            flags=re.IGNORECASE,
        )
        x = re.sub(r"Created by MOE:.*?\nMOE_MIGRATED_REVID=.*?($|\n)", "", x)
        x = re.sub(
            r"(fbshipit-source-id|Differential Revision|Change-Id|PiperOrigin-RevId|BAZEL_VERSION_REV_ID).*?($|\n)",
            "",
            x,
            flags=re.IGNORECASE,
        )
        x = re.sub(r"(BUG=|FIXED=)\d*?($|\n)", "", x)
        return x

    @staticmethod
    def _is_trivial_or_bot(message: str) -> bool:
        message = message.strip()
        # pad punctuation with spaces - expected format in given regular expressions
        message = message.translate(str.maketrans({key: " {0} ".format(key) for key in punctuation}))
        message = re.sub(" +", " ", message)

        patterns = [
            # for bot messages
            r"^ignore update \' .* \.$",
            # for shadow messages
            r"^update(d)? (changelog|gitignore|readme( . md| file)?)( \.)?$",
            r"^prepare version (v)?[ \d.]+$",
            r"^bump (up )?version( number| code)?( to (v)?[ \d.]+( - snapshot)?)?( \.)?$",
            r"^modify (dockerfile|makefile)( \.)?$",
            r"^update submodule(s)?( \.)?$",
        ]

        for pattern in patterns:
            if re.match(pattern, message, flags=re.IGNORECASE):
                return True

        return False

    @staticmethod
    def _filter(message: str) -> str:
        if not isinstance(message, str) or not message.isascii() or MessageProcessor._is_trivial_or_bot(message):
            return ""

        x = MessageProcessor._filter_emails(message)
        x = MessageProcessor._filter_urls(x)
        x = MessageProcessor._filter_issue_ref(x)
        x = MessageProcessor._filter_signature(x)
        x = MessageProcessor._filter_at_pattern(x)
        x = MessageProcessor._filter_sha(x)
        x = x.replace("\n", " ")
        x = x.strip()
        return x

    def process(self, chunk: pd.DataFrame, **kwargs) -> pd.DataFrame:
        with Parallel(self._n_workers) as pool:
            filtered_messages = pool(
                delayed(MessageProcessor._filter)(message) for _, message in chunk["message"].items()
            )

        chunk["message"] = filtered_messages
        return chunk.loc[chunk.message.str.len() > 0]
