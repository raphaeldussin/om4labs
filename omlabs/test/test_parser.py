def test_default_diag_parser():
    from om4labs.om4parser import default_diag_parser
    from om4labs.om4parser import DefaultDictParser

    parser = default_diag_parser(template=True)
    args = parser.parse_args([])
    args = args.__dict__
    num_default_options = len(list(args.keys()))
    assert isinstance(args, dict), "unable to obtain dict of options"
    assert num_default_options > 0, "no options are present"

    parser = default_diag_parser(template=False)
    args = parser.parse_args(["foo"])
    _num_keys = len(list(args.__dict__.keys()))
    assert (
        _num_keys == num_default_options
    ), "argparse object does not have the same number of keys"

    parser = default_diag_parser(template=False, exclude="gridspec")
    args = parser.parse_args(["foo"])
    _num_keys = len(list(args.__dict__.keys()))
    assert _num_keys == (num_default_options - 1), "excluding a single string failed"

    parser = default_diag_parser(template=False, exclude=["gridspec", "basin"])
    args = parser.parse_args(["foo"])
    _num_keys = len(list(args.__dict__.keys()))
    assert _num_keys == (num_default_options - 2), "excluding a list failed"
