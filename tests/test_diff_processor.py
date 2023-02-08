from src.processing import DiffProcessor


def test_filter_diff():
    diff = "@@ ..,.. @@\ncontext1\ncontext2\ncontext3\n-remove\n+add\nBinary files x and y differ\ncontext4\ncontext5\ncontext6\n"
    assert (
        DiffProcessor._process_diff(diff, line_sep="\n")
        == "context1\ncontext2\ncontext3\n-remove\n+add\nBinary files x and y differ\ncontext4\ncontext5\ncontext6\n"
    )

    diff = "@@ ..,.. @@\n@decorator\n-def func():\n-    do_smth()\n\ No newline at end of file\n+def func():\n+\tdo_smth()\n\n\n\n"
    assert (
        DiffProcessor._process_diff(diff, line_sep="\n")
        == "@decorator\n-def func():\n- do_smth()\n\ No newline at end of file\n+def func():\n+ do_smth()\n"
    )
