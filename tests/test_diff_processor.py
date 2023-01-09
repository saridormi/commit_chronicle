from src.processing import DiffProcessor


def test_filter_diff():
    diff = "context1\ncontext2\ncontext3\n-remove\n+add\nBinary files x and y differ\ncontext4\ncontext5\ncontext6\n"
    assert DiffProcessor._filter_diff(diff, line_sep="\n") == "-remove\n+add\nBinary files x and y differ\n"

    diff = "-remove\n\ No newline at end of file\n+add\n \n\n\n\n"
    assert DiffProcessor._filter_diff(diff, line_sep="[NL]") == "-remove[NL]+add[NL]"
