from secrets import token_hex
from typing import Callable, List

import pytest

from src.processing import MessageProcessor


@pytest.fixture
def patterns():
    return dict(
        emails=["example.exa-mple_exampLe123@mail.com"],
        urls=[
            "https://smth.com/",
            "http://smth.com/?smth=else&smth=else#smth",
            "https://github.com/total-typescript/beginners-typescript-tutorial",
        ],
        at_patterns=["@simplenickname", "@@", "@nick_Name9999"],
        issue_refs=["#123", "GH-123", "gh-123", "SMTH-123"],
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


def generic_test_punctuation_end(method: Callable, patterns: List[str], **kwargs):
    for pattern in patterns:
        for punctuation in [".", "?", "........", "???", "!!", "~", ")"]:
            cur_pattern = pattern + punctuation
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


def generic_test_punctuation_start(method: Callable, patterns: List[str], **kwargs):
    for pattern in patterns:
        for punctuation in ["- ", ": ", "-", ":"]:
            cur_pattern = punctuation + pattern
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


def test_filter_emails(patterns):
    emails = patterns["emails"]
    generic_test(MessageProcessor._filter_emails, emails)


def test_filter_urls(patterns):
    urls = patterns["urls"]
    generic_test(MessageProcessor._filter_urls, urls)


def test_filter_at_pattern(patterns):
    at_patterns = patterns["at_patterns"]
    generic_test(MessageProcessor._filter_at_pattern, at_patterns)
    assert MessageProcessor._filter_at_pattern("name@mail.com") == "name@mail.com"


def test_filter_sha():
    # generate random hashes 100 times
    for _ in range(100):
        shas = [f"{sha_prefix}{token_hex(nbytes=20)}" for sha_prefix in ["", "ref:", "I"]]
        generic_test(MessageProcessor._filter_sha, shas)

    # unfortunately, there are some false positives
    with pytest.raises(AssertionError):
        assert MessageProcessor._filter_sha("deedeed")
        assert MessageProcessor._filter_sha("acceded")
        assert MessageProcessor._filter_sha("1009000")


def test_filter_issue_ref(patterns):
    issue_refs = patterns["issue_refs"]
    generic_test(MessageProcessor._filter_issue_ref, issue_refs)

    # unfortunately, there are false positives when filtering JIRA projects ids
    with pytest.raises(AssertionError):
        assert MessageProcessor._filter_issue_ref("GPT-2")


def test_filter_signature():
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
        assert not MessageProcessor._filter_signature(f"{signature} some right context")
        # in the middle
        assert (
            MessageProcessor._filter_signature(f"some left context {signature} some right context")
            == "some left context"
        )
        # in the end
        assert MessageProcessor._filter_signature(f"some left context {signature}") == "some left context"
        # the whole line
        assert not MessageProcessor._filter_signature(f"{signature})")

    beginning_only_signatures = [
        "Bug:",
        "BUG=",
        "FIXES=",
        "R=",
    ]
    for signature in beginning_only_signatures:
        # in the beginning
        assert not MessageProcessor._filter_signature(f"{signature} some right context")
        # the whole line
        assert not MessageProcessor._filter_signature(f"{signature})")


def test_is_trivial_or_bot():
    # bot message
    assert MessageProcessor._is_trivial_or_bot("ignore update 'anything.")
    assert not MessageProcessor._is_trivial_or_bot("ignore update 'anything. but it is not the whole message")
    assert not MessageProcessor._is_trivial_or_bot("ignore update 'anything.\nbut it is not the whole message")

    # trivial message – update(d) readme/gitignore/changelog
    for upd_option in ["", "d"]:
        for fname_option in ["readme", "readme.md", "readme file", "gitignore", "changelog"]:
            assert MessageProcessor._is_trivial_or_bot(f"update{upd_option} {fname_option}")
    assert not MessageProcessor._is_trivial_or_bot("update readme but it is not the whole message")
    assert not MessageProcessor._is_trivial_or_bot("update readme\nbut it is not the whole message")

    # trivial message – prepare/bump version
    assert MessageProcessor._is_trivial_or_bot("prepare version 123123")
    assert MessageProcessor._is_trivial_or_bot("bump version to v1.2")
    assert MessageProcessor._is_trivial_or_bot("bump up version code.")
    assert MessageProcessor._is_trivial_or_bot("bump version number to v1.2.3-SNAPSHOT")
    assert not MessageProcessor._is_trivial_or_bot("bump up version but it is not the whole message")
    assert not MessageProcessor._is_trivial_or_bot("bump up version\nbut it is not the whole message")

    # trivial message – modily Dockerfile/Makefile
    assert MessageProcessor._is_trivial_or_bot("modify dockerfile.")
    assert MessageProcessor._is_trivial_or_bot("Modify Makefile.")
    assert not MessageProcessor._is_trivial_or_bot("modify makefile but it is not the whole message")
    assert not MessageProcessor._is_trivial_or_bot("modify makefile\nbut it is not the whole message")

    # trivial message – update submodule(s)
    assert MessageProcessor._is_trivial_or_bot("update submodule.")
    assert MessageProcessor._is_trivial_or_bot("update submodules")
    assert not MessageProcessor._is_trivial_or_bot("update submodule but it is not the whole message")
    assert not MessageProcessor._is_trivial_or_bot("update submodule\nbut it is not the whole message")


def test_filter(patterns):
    for patterns_type in patterns:
        generic_test(MessageProcessor._filter, patterns[patterns_type], line_sep="\n")
        generic_test_punctuation_end(MessageProcessor._filter, patterns[patterns_type], line_sep="\n")
        generic_test_punctuation_start(MessageProcessor._filter, patterns[patterns_type], line_sep="\n")

    assert MessageProcessor._filter("[#123] - meaningful message", line_sep="_") == "meaningful message"
    assert MessageProcessor._filter("Fix [#123]", line_sep="_") == "Fix"
    assert MessageProcessor._filter("[PROJECTNAME-123] - Meaningful message", line_sep="_") == "Meaningful message"
    assert MessageProcessor._filter("* PROJECTNAME-123: meaningful message", line_sep="_") == "* meaningful message"
    assert (
        MessageProcessor._filter("[Some kind of tag] Meaningful message", line_sep="_")
        == "[Some kind of tag] Meaningful message"
    )

    assert MessageProcessor._filter("1970-01-01 00:00:00", line_sep="_") == "1970-01-01 00:00:00"
    assert MessageProcessor._filter("01-01-1970 00:00:00", line_sep="_") == "01-01-1970 00:00:00"

    non_ascii = "下午好"
    assert not MessageProcessor._filter(non_ascii, line_sep="_")

    example = "line1\nline2"
    assert MessageProcessor._filter(example, line_sep="[NL]") == "[NL]".join(example.split("\n"))
