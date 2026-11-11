from nexttrack import constants


def test_split_boundaries_match_official_protocol():
    # Values from benchmarks/yambda/constants.py and timesplit.py.
    assert constants.TEST_START == 26_000_000 - 86_400
    assert constants.VAL_END == constants.TEST_START - 1_800
    assert constants.VAL_START == constants.VAL_END - 86_400
    assert constants.TRAIN_END == constants.VAL_START - 1_800
    assert constants.TRAIN_END == 25_823_600


def test_split_is_ordered_with_gaps():
    assert 0 < constants.TRAIN_END < constants.VAL_START
    assert constants.VAL_START < constants.VAL_END < constants.TEST_START
    assert constants.VAL_START - constants.TRAIN_END == constants.GAP_SECONDS
    assert constants.TEST_START - constants.VAL_END == constants.GAP_SECONDS


def test_ranking_constants():
    assert constants.NUM_RANKED == max(constants.K_VALUES)
    assert constants.NUM_CANDIDATES >= constants.NUM_RANKED
    assert constants.PADDING_INDEX == 0
    assert constants.LISTEN_THRESHOLD_PCT == 50
