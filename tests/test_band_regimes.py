import pandas as pd
import pytest
from src.band_regimes import attach_limits, floor_threshold, regime_table


def test_floor_thresholds_match_psx_notice():
    assert floor_threshold(0.075) == pytest.approx(13.33, abs=0.01)  # PSX 2020 notice
    assert floor_threshold(0.070) == pytest.approx(14.29, abs=0.01)
    assert floor_threshold(0.05) == pytest.approx(20.0)
    assert floor_threshold(0.10) == pytest.approx(10.0)


def test_regimes_sorted_and_monotone():
    t = regime_table()
    assert t["start"].is_monotonic_increasing
    assert t["pct"].is_monotonic_increasing


def _px(dates, closes):
    return pd.DataFrame({"date": pd.to_datetime(dates), "close": closes, "volume": 1})


def test_upper_limit_hit_pct_regime():
    f = attach_limits(_px(["2025-01-02", "2025-01-03"], [100.0, 110.0]), tick=0.01)
    assert f["censored_up"].iat[0] and not f["floor_binds"].iat[0]


def test_floor_regime_uses_re1():
    f = attach_limits(_px(["2025-01-02", "2025-01-03"], [5.0, 6.0]), tick=0.01)
    assert f["floor_binds"].iat[0] and f["censored_up"].iat[0]


def test_breach_not_counted_as_censored():
    # 50% drop = bonus/split ex-date, not a limit hit
    f = attach_limits(_px(["2025-01-02", "2025-01-03"], [100.0, 50.0]), tick=0.01)
    assert f["breach"].iat[0] and not f["censored"].iat[0]


def test_regime_switch_2024():
    f = attach_limits(_px(["2024-07-19", "2024-07-22"], [100.0, 110.0]), tick=0.01)
    assert f["pct"].iat[0] == 0.10 and f["censored_up"].iat[0]

def test_unsorted_and_duplicate_dates():
    f = attach_limits(_px(["2025-01-03", "2025-01-02", "2025-01-03"], [110.0, 100.0, 110.0]))
    assert len(f) == 1 and f["censored_up"].iat[0]


# TODO once raw data exists: ISL replication
# def test_isl_replication():
#     from scipy.stats import chi2_contingency
#     f = attach_limits(pd.read_parquet("data/raw/ISL.parquet"))
#     ct = pd.crosstab(f["floor_binds"], f["censored"])
#     p = chi2_contingency(ct)[1]
#     print(p)  # compare to 3.7e-17 from the 30-stock version
