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
    def _filter_generic(x: str, pattern: str):
        x = re.sub(r"^[:\-\s]*" + pattern + "[:,.!?;~)]*(\s|$)", "", x)
        x = re.sub(r"\s[:\-\s]*" + pattern + "[:,.!?;~)]*(?=\s|$)", "", x)
        return x

    @staticmethod
    def _filter_emails(message: str) -> str:
        email_pattern = r"[a-zA-Z0-9][\w.\-+]*@([\w.\-])+\.[\w.\-]+"
        email_pattern = r"[\[\({<]?" + email_pattern + r"[\]\)}>]?"
        return MessageProcessor._filter_generic(message, email_pattern)

    @staticmethod
    def _filter_urls(message: str) -> str:
        url_pattern = r"https?://[-a-zA-Z0-9@:%._+~#?=/&]+"
        return MessageProcessor._filter_generic(message, url_pattern)

    @staticmethod
    def _filter_at_pattern(message: str) -> str:
        at_pattern = r"@\S+"
        return MessageProcessor._filter_generic(message, at_pattern)

    @staticmethod
    def _filter_sha(message: str) -> str:
        x = message
        # trying to avoid false positives
        if re.search("\d\d\d\d-\d\d-\d\d", x) or re.search("\d\d-\d\d-\d\d\d\d", x):
            return x

        for sha_prefix in ["", "ref:", "I"]:
            x = MessageProcessor._filter_generic(x, sha_prefix + r"[\dA-Fa-f-]{7,}")
        return x

    @staticmethod
    def _filter_issue_ref(message: str) -> str:
        """
        Deletes issue numbers from the following patterns:
        * #123
        * GH-123
        * gh-123
        * ANYTHING-123 (Jira project id)
        """
        x = message
        for pattern in [r"#[\d]+", r"GH-[\d]+", r"gh-[\d]+", r"([A-Z][A-Z0-9]+-[0-9]+)"]:
            pattern = r"[\[\({<]?" + pattern + r"[\]\)}>]?"
            x = MessageProcessor._filter_generic(x, pattern)
            x = x.lstrip("-–:")
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
            * `git-svn-id: <url>`
            * `Bug: <number>`
            * `Reviewed-on: <url>`
            * `Auto-Submit: <username>`
            * `Commit-Queue: <username>`
        * https://github.com/google/moe
            * `Created by MOE: <some link>`
            * `MOE_MIGRATED_REVID=<some number>`
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
        * https://github.com/kubernetes/sample-apiserver
            * `Kubernetes-commit: <id>`
        * https://github.com/catboost/catboost
            * `Revision: r<number>`
            * `Sandbox task ID: <id>`
            * `Glycine run ID: <id>`
        * https://github.com/luci/luci-go
            * `R=emails/nicknames`
        """
        x = message
        for pattern in [
            r"(signed(-| |)off(-| |)by|co(-| |)?authored(-| |)by|also(-| |)by|reviewed?(-| |)(by|on)|former(-| |)commit(-| |)id|git-svn-id|auto-submit|commit-queue)",
            r"(Created by MOE|MOE_MIGRATED_REVID)",
            r"(fbshipit-source-id|Differential Revision|Pulled(-| )by)",
            r"(Change-Id|PiperOrigin-RevId|BAZEL_VERSION_REV_ID)",
            r"(Kubernetes-commit)",
            r"(Revision: r[\d]*|Sandbox task ID|Glycine run ID)",
        ]:
            x = re.sub(
                r"(^|\s)" + pattern + r".*?$",
                "",
                x,
                flags=re.IGNORECASE,
            )
        #
        for pattern in [r"(Bug:|BUG=|FIXES=|R=)"]:
            x = re.sub(
                r"^" + pattern + r".*?$",
                "",
                x,
            )
        return x

    @staticmethod
    def _is_trivial_or_bot(message: str) -> bool:
        message = message.strip()
        # pad punctuation with spaces - expected format in given regular expressions
        message = message.translate(str.maketrans({key: " {0} ".format(key) for key in punctuation}))
        message = re.sub(" +", " ", message).strip()

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
    def _filter(message: str, line_sep: str) -> str:
        if not isinstance(message, str) or not message.isascii() or MessageProcessor._is_trivial_or_bot(message):
            return ""
        message_lines = message.split("\n")
        for i, line in enumerate(message_lines):
            line = MessageProcessor._filter_emails(line)
            line = MessageProcessor._filter_urls(line)
            line = MessageProcessor._filter_issue_ref(line)
            line = MessageProcessor._filter_signature(line)
            line = MessageProcessor._filter_at_pattern(line)
            line = MessageProcessor._filter_sha(line)
            line = line.strip()
            message_lines[i] = line

        return line_sep.join([line for line in message_lines if line])

    def process(self, chunk: pd.DataFrame, line_sep: str, **kwargs) -> pd.DataFrame:
        with Parallel(self._n_workers) as pool:
            filtered_messages = pool(
                delayed(MessageProcessor._filter)(message, line_sep) for _, message in chunk["message"].items()
            )

        chunk["message"] = filtered_messages
        return chunk.loc[chunk.message.str.len() > 0]
