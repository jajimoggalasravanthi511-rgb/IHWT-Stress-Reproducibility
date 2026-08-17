import numpy as np
from sklearn.model_selection import LeaveOneGroupOut


def test_loso_has_no_subject_overlap():
    groups = np.repeat(["S1", "S2", "S3", "S4"], 10)
    X = np.zeros((40, 2)); y = np.tile([0, 1], 20)
    for tr, te in LeaveOneGroupOut().split(X, y, groups):
        assert set(groups[tr]).isdisjoint(set(groups[te]))
        assert len(set(groups[te])) == 1
