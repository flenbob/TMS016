from dataclasses import dataclass
from sklearn.ensemble import AdaBoostClassifier
import numpy as np

@dataclass
class Model:
    __slots__ = ['clf', 'threshold', 'feat_types', 'feats', 'feats_idx']
    clf: AdaBoostClassifier
    threshold: float
    feat_types: np.float64
    feats: np.ndarray
    feats_idx: np.ndarray