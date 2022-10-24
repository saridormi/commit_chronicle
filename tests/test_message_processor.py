from secrets import token_hex
from typing import Callable, List

import pytest

from src.processing import MessageProcessor


@pytest.fixture
def patterns():
    return dict(
        email=["example.exa-mple_exampLe123@mail.com"],
        url=[
            "https://smth.com/",
            "http://smth.com/?smth=else&smth=else#smth",
            "https://github.com/total-typescript/beginners-typescript-tutorial",
        ],
        at_pattern=["@simplenickname", "@@", "@nick_Name9999"],
        issue_ref=["#123", "GH-123", "gh-123", "SMTH-123"],
    )


def generic_test(method: Callable, patterns: List[str], **kwargs):
    for cur_pattern in patterns:
        # in the beginning
        assert method(f"{cur_pattern} some right context", **kwargs) == "some right context"
        # in the middle
        assert (
            method(f"some left context {cur_pattern} some right context", **kwargs)
            == f"some left context some right context"
        )
        # in the end
        assert method(f"some left context {cur_pattern}", **kwargs) == "some left context"
        # the whole line
        assert not method(cur_pattern, **kwargs)


def generic_test_replace_patterns(method: Callable, patterns: List[str], replace_special_token: str, **kwargs):
    for cur_pattern in patterns:
        # in the beginning
        assert method(f"{cur_pattern} some right context", **kwargs) == f"{replace_special_token} some right context"
        # in the middle
        assert (
            method(f"some left context {cur_pattern} some right context", **kwargs)
            == f"some left context {replace_special_token} some right context"
        )
        # in the end
        assert method(f"some left context {cur_pattern}", **kwargs) == f"some left context {replace_special_token}"
        # the whole line
        assert not method(cur_pattern, **kwargs)


def generic_test_punctuation_end(method: Callable, patterns: List[str], **kwargs):
    for pattern in patterns:
        for punctuation in [".", "?", "........", "???", "!!", "~"]:
            # in the beginning
            assert method(f"{pattern}{punctuation} some right context", **kwargs) == "some right context"
            # in the middle
            assert (
                method(f"some left context {pattern}{punctuation} some right context", **kwargs)
                == f"some left context some right context"
            )
            # in the end
            assert method(f"some left context {pattern}{punctuation}", **kwargs) == "some left context"
            # the whole line
            assert not method(f"{pattern}{punctuation}", **kwargs)


def generic_test_punctuation_end_replace_patterns(
    method: Callable, patterns: List[str], replace_special_token: str, **kwargs
):
    for pattern in patterns:
        for punctuation in [".", "?", "........", "???", "!!", "~"]:
            # in the beginning
            assert (
                method(f"{pattern}{punctuation} some right context", **kwargs)
                == f"{replace_special_token}{punctuation} some right context"
            )
            # in the middle
            assert (
                method(f"some left context {pattern}{punctuation} some right context", **kwargs)
                == f"some left context {replace_special_token}{punctuation} some right context"
            )
            # in the end
            assert (
                method(f"some left context {pattern}{punctuation}", **kwargs)
                == f"some left context {replace_special_token}{punctuation}"
            )
            # the whole line
            assert not method(f"{pattern}{punctuation}", **kwargs)


def generic_test_punctuation_start(method: Callable, patterns: List[str], **kwargs):
    for pattern in patterns:
        for punctuation in ["- ", ": ", "-", ":"]:
            # in the beginning
            assert method(f"{punctuation}{pattern} some right context", **kwargs) == "some right context"
            # in the middle
            assert (
                method(f"some left context {punctuation}{pattern} some right context", **kwargs)
                == f"some left context some right context"
            )
            # in the end
            assert method(f"some left context {punctuation}{pattern}", **kwargs) == "some left context"
            # the whole line
            assert not method(f"{punctuation}{pattern}", **kwargs)


def generic_test_punctuation_start_replace_patterns(
    method: Callable, patterns: List[str], replace_special_token: str, **kwargs
):
    for pattern in patterns:
        for punctuation in ["- ", ": ", "-", ":"]:
            # in the beginning
            assert (
                method(f"{punctuation}{pattern} some right context", **kwargs)
                == f"{punctuation}{replace_special_token} some right context"
            )
            # in the middle
            assert (
                method(f"some left context {punctuation}{pattern} some right context", **kwargs)
                == f"some left context {punctuation}{replace_special_token} some right context"
            )
            # in the end
            assert (
                method(f"some left context {punctuation}{pattern}", **kwargs)
                == f"some left context {punctuation}{replace_special_token}"
            )
            # the whole line
            assert (
                not method(f"{punctuation}{pattern}", **kwargs)
                if len(f"{punctuation}{pattern}".split()) == 1
                else f"{punctuation}{replace_special_token}"
            )


def test_remove_emails(patterns):
    emails = patterns["email"]
    generic_test(MessageProcessor._remove_emails, emails)


def test_remove_urls(patterns):
    urls = patterns["url"]
    generic_test(MessageProcessor._remove_urls, urls)


def test_remove_at_pattern(patterns):
    at_patterns = patterns["at_pattern"]
    generic_test(MessageProcessor._remove_at_pattern, at_patterns)
    assert MessageProcessor._remove_at_pattern("name@mail.com") == "name@mail.com"


def test_remove_sha():
    # generate random hashes 100 times
    for _ in range(100):
        shas = [f"{sha_prefix}{token_hex(nbytes=20)}" for sha_prefix in ["", "ref:", "I"]]
        generic_test(MessageProcessor._remove_sha, shas)

    # unfortunately, there are some false positives
    with pytest.raises(AssertionError):
        assert MessageProcessor._remove_sha("deedeed")
        assert MessageProcessor._remove_sha("acceded")
        assert MessageProcessor._remove_sha("1009000")


def test_remove_issue_ref(patterns):
    issue_refs = patterns["issue_ref"]
    generic_test(MessageProcessor._remove_issue_ref, issue_refs)

    # unfortunately, there are false positives when filtering JIRA projects ids
    with pytest.raises(AssertionError):
        assert MessageProcessor._remove_issue_ref("GPT-2")


def test_remove_signature():
    signatures = (
        [f"signed{sep1}off{sep2}by" for sep1 in [" ", "-"] for sep2 in [" ", "-"]]
        + [f"co{sep1}authored{sep2}by" for sep1 in [" ", "-", ""] for sep2 in [" ", "-"]]
        + [
            "Also by",
            "Reviewed By",
            "Former commit id",
            "git-svn-id",
            "Reviewed-On",
            "Auto-Submit",
            "Commit-Queue",
            "Differential Revision",
            "Pulled By",
            "fbshipit-source-id",
            "Created by MOE",
            "MOE_MIGRATED_REVID",
            "Change-Id",
            "PiperOrigin-RevId",
            "BAZEL_VERSION_REV_ID",
            "Kubernetes-commit",
            "Revision: r",
            "Sandbox task ID",
            "Glycine run ID",
        ]
    )
    for signature in signatures:
        # in the beginning
        assert not MessageProcessor._remove_signatures(f"{signature} some right context")
        # in the middle
        assert (
            MessageProcessor._remove_signatures(f"some left context {signature} some right context")
            == "some left context"
        )
        # in the end
        assert MessageProcessor._remove_signatures(f"some left context {signature}") == "some left context"
        # the whole line
        assert not MessageProcessor._remove_signatures(f"{signature})")

    beginning_only_signatures = [
        "Bug:",
        "BUG=",
        "FIXES=",
        "R=",
    ]
    for signature in beginning_only_signatures:
        # in the beginning
        assert not MessageProcessor._remove_signatures(f"{signature} some right context")
        # the whole line
        assert not MessageProcessor._remove_signatures(f"{signature})")


def test_filter_trivial_or_bot():
    # bot message
    assert MessageProcessor._filter_trivial_or_bot("ignore update 'anything.")
    assert not MessageProcessor._filter_trivial_or_bot("ignore update 'anything. but it is not the whole message")
    assert not MessageProcessor._filter_trivial_or_bot("ignore update 'anything.\nbut it is not the whole message")

    # trivial message – update(d) readme/gitignore/changelog
    for upd_option in ["", "d"]:
        for fname_option in ["readme", "readme.md", "readme file", "gitignore", "changelog"]:
            assert MessageProcessor._filter_trivial_or_bot(f"update{upd_option} {fname_option}")
    assert not MessageProcessor._filter_trivial_or_bot("update readme but it is not the whole message")
    assert not MessageProcessor._filter_trivial_or_bot("update readme\nbut it is not the whole message")

    # trivial message – prepare/bump version
    assert MessageProcessor._filter_trivial_or_bot("prepare version 123123")
    assert MessageProcessor._filter_trivial_or_bot("bump version to v1.2")
    assert MessageProcessor._filter_trivial_or_bot("bump up version code.")
    assert MessageProcessor._filter_trivial_or_bot("bump version number to v1.2.3-SNAPSHOT")
    assert not MessageProcessor._filter_trivial_or_bot("bump up version but it is not the whole message")
    assert not MessageProcessor._filter_trivial_or_bot("bump up version\nbut it is not the whole message")

    # trivial message – modily Dockerfile/Makefile
    assert MessageProcessor._filter_trivial_or_bot("modify dockerfile.")
    assert MessageProcessor._filter_trivial_or_bot("Modify Makefile.")
    assert not MessageProcessor._filter_trivial_or_bot("modify makefile but it is not the whole message")
    assert not MessageProcessor._filter_trivial_or_bot("modify makefile\nbut it is not the whole message")

    # trivial message – update submodule(s)
    assert MessageProcessor._filter_trivial_or_bot("update submodule.")
    assert MessageProcessor._filter_trivial_or_bot("update submodules")
    assert not MessageProcessor._filter_trivial_or_bot("update submodule but it is not the whole message")
    assert not MessageProcessor._filter_trivial_or_bot("update submodule\nbut it is not the whole message")

    # trivial message – update any_file_name.any_file_extension
    assert MessageProcessor._filter_trivial_or_bot("update SoMetHiNg123.someTHING")


def test_filter_merge():
    assert MessageProcessor._filter_merge("Merge branch xxx")
    assert MessageProcessor._filter_merge("Merge pull request xxx")

    # only capitalized merge will be filtered
    assert not MessageProcessor._filter_merge("merge xxx")

    # unfortunately, there are some false positives
    assert MessageProcessor._filter_merge("Merge something completely unrelated to merge commits")


def test_filter_revert():
    assert MessageProcessor._filter_revert('Revert "old commit message"')

    # only capitalized revert will be filtered
    assert not MessageProcessor._filter_revert("revert xxx")

    # unfortunately, there are some false positives
    assert MessageProcessor._filter_revert("Revert something completely unrelated to revert commits")


def test_filter_squash():
    assert MessageProcessor._filter_squash("* Commit message 1\n* Commit message 2\n* Commit message 3", line_sep="\n")
    assert MessageProcessor._filter_squash(
        "PR Title\n* Commit message 1\n* Commit message 2\n* Commit message 3", line_sep="\n"
    )

    # unfortunately, there are some false positives
    assert MessageProcessor._filter_squash(
        "Maybe it's the summary\n* Maybe it's some detail about the change\n* Maybe it's another detail about the change",
        line_sep="\n",
    )


def test_process(patterns):
    for patterns_type in patterns:
        generic_test(MessageProcessor._process, patterns[patterns_type], line_sep="\n", replace_patterns=False)
        generic_test_punctuation_end(
            MessageProcessor._process, patterns[patterns_type], line_sep="\n", replace_patterns=False
        )
        generic_test_punctuation_start(
            MessageProcessor._process, patterns[patterns_type], line_sep="\n", replace_patterns=False
        )

    assert (
        MessageProcessor._process("[#123] - meaningful message", line_sep="_", replace_patterns=False)
        == "meaningful message"
    )
    assert (
        MessageProcessor._process("Meaningful message\nFix [#123]", line_sep="\n", replace_patterns=False)
        == "Meaningful message\nFix"
    )
    assert (
        MessageProcessor._process("[PROJECTNAME-123] - Meaningful message", line_sep="_", replace_patterns=False)
        == "Meaningful message"
    )
    assert (
        MessageProcessor._process("* PROJECTNAME-123: meaningful message", line_sep="_", replace_patterns=False)
        == "* meaningful message"
    )
    assert (
        MessageProcessor._process("[Some kind of tag] Meaningful message", line_sep="_", replace_patterns=False)
        == "[Some kind of tag] Meaningful message"
    )

    assert (
        MessageProcessor._process("1970-01-01 00:00:00", line_sep="_", replace_patterns=False) == "1970-01-01 00:00:00"
    )
    assert (
        MessageProcessor._process("01-01-1970 00:00:00", line_sep="_", replace_patterns=False) == "01-01-1970 00:00:00"
    )

    non_ascii = "下午好"
    assert not MessageProcessor._process(non_ascii, line_sep="_", replace_patterns=False)

    example = "message line 1\nmessage line 2"
    assert MessageProcessor._process(example, line_sep="[NL]", replace_patterns=False) == "[NL]".join(
        example.split("\n")
    )


def test_process_replace_patterns(patterns):
    for patterns_type in patterns:
        generic_test_replace_patterns(
            MessageProcessor._process,
            patterns[patterns_type],
            replace_special_token=MessageProcessor.get_special_tokens()[patterns_type],
            line_sep="\n",
            replace_patterns=True,
        )
        generic_test_punctuation_end_replace_patterns(
            MessageProcessor._process,
            patterns[patterns_type],
            replace_special_token=MessageProcessor.get_special_tokens()[patterns_type],
            line_sep="\n",
            replace_patterns=True,
        )
        generic_test_punctuation_start_replace_patterns(
            MessageProcessor._process,
            patterns[patterns_type],
            replace_special_token=MessageProcessor.get_special_tokens()[patterns_type],
            line_sep="\n",
            replace_patterns=True,
        )

    assert (
        MessageProcessor._process("[#123] - meaningful message", line_sep="_", replace_patterns=True)
        == "[ISSUE_ID] - meaningful message"
    )
    assert (
        MessageProcessor._process("Meaningful message\nFix #123", line_sep="\n", replace_patterns=True)
        == "Meaningful message\nFix [ISSUE_ID]"
    )
    assert (
        MessageProcessor._process("[PROJECTNAME-123] - Meaningful message", line_sep="_", replace_patterns=True)
        == "[ISSUE_ID] - Meaningful message"
    )
    assert (
        MessageProcessor._process("* PROJECTNAME-123: meaningful message", line_sep="_", replace_patterns=True)
        == "* [ISSUE_ID]: meaningful message"
    )
    assert (
        MessageProcessor._process("[Some kind of tag] Meaningful message", line_sep="_", replace_patterns=True)
        == "[Some kind of tag] Meaningful message"
    )

    assert (
        MessageProcessor._process("1970-01-01 00:00:00", line_sep="_", replace_patterns=True) == "1970-01-01 00:00:00"
    )
    assert (
        MessageProcessor._process("01-01-1970 00:00:00", line_sep="_", replace_patterns=True) == "01-01-1970 00:00:00"
    )

    non_ascii = "下午好"
    assert not MessageProcessor._process(non_ascii, line_sep="_", replace_patterns=True)

    example = "message line 1\nmessage line 2"
    assert MessageProcessor._process(example, line_sep="[NL]", replace_patterns=True) == "[NL]".join(
        example.split("\n")
    )
