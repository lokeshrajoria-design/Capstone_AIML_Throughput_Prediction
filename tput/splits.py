"""Step 4: trace-level evaluation partitions, fixed before any model is fitted.

  outer_fold  - 5 folds of whole traces, stratified by application x mobility (primary evaluation)
  date_fold   - whole recording dates held out together (date-dependence sensitivity)
  chrono_test - traces from the latest dates held out (chronological sensitivity)
Inner 3-fold model selection uses inner_folds() on the outer training traces only.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

from . import config as C


def make_splits(manifest, eligible_traces):
    m = manifest[manifest.trace_id.isin(eligible_traces)].reset_index(drop=True)
    strata = m.application_type + "|" + m.mobility_condition

    outer = StratifiedGroupKFold(n_splits=C.OUTER_FOLDS, shuffle=True, random_state=C.SEED)
    m["outer_fold"] = -1
    for k, (_, test) in enumerate(outer.split(m, strata, groups=m.trace_id)):
        m.loc[test, "outer_fold"] = k

    m["date_fold"] = -1
    for k, (_, test) in enumerate(GroupKFold(n_splits=C.OUTER_FOLDS).split(m, groups=m.recording_date)):
        m.loc[test, "date_fold"] = k

    dates = np.sort(m.recording_date.unique())
    cutoff = dates[int(np.ceil(0.8 * len(dates)))]          # latest ~20 % of recording dates
    m["chrono_test"] = (m.recording_date >= cutoff).astype(int)
    return m[["trace_id", "application_type", "mobility_condition", "recording_date",
              "outer_fold", "date_fold", "chrono_test"]]


def inner_folds(train_traces, strata):
    """Inner grouped folds for model selection inside one outer training partition."""
    cv = StratifiedGroupKFold(n_splits=C.INNER_FOLDS, shuffle=True, random_state=C.SEED)
    return list(cv.split(train_traces, strata, groups=train_traces))
