import os
import argparse
from settings.hp_grid import HP_MINIBATCH_SIZE
import pandas as pd
import numpy as np
from functools import reduce

from settings.default import (
    QUANDL_TICKERS,
    FEATURES_OPENBB_FILE_PATH,
)
from settings.fixed_params import MODLE_PARAMS
from mom_trans.backtest import run_all_windows

# define the asset class of each ticker here
TEST_MODE = False
ASSET_CLASS_MAPPING = dict(zip(QUANDL_TICKERS, ["COMB"] * len(QUANDL_TICKERS)))
TRAIN_VALID_RATIO = 0.90
TIME_FEATURES = False
FORCE_OUTPUT_SHARPE_LENGTH = None
EVALUATE_DIVERSIFIED_VAL_SHARPE = True
NAME = "experiment_openbb_100assets"  # you may rename this prefix


def main(
    experiment: str,
    train_start: int,
    test_start: int,
    test_end: int,
    test_window_size: int,
    num_repeats: int,
):
    # ------------------------------------------------------------
    # pick architecture & CPD lookback(s)
    # ------------------------------------------------------------
    if experiment == "LSTM":
        architecture = "LSTM"
        total_time_steps = 63
        changepoint_lbws = None
    elif experiment == "LSTM-CPD-21":
        architecture = "LSTM"
        total_time_steps = 63
        changepoint_lbws = [21]
    elif experiment == "LSTM-CPD-63":
        architecture = "LSTM"
        total_time_steps = 63
        changepoint_lbws = [63]
    elif experiment == "TFT":
        architecture = "TFT"
        total_time_steps = 252
        changepoint_lbws = None
    elif experiment == "TFT-CPD-126-21":
        architecture = "TFT"
        total_time_steps = 252
        changepoint_lbws = [126, 21]
    elif experiment == "TFT-SHORT":
        architecture = "TFT"
        total_time_steps = 63
        changepoint_lbws = None
    elif experiment == "TFT-SHORT-CPD-21":
        architecture = "TFT"
        total_time_steps = 63
        changepoint_lbws = [21]
    elif experiment == "TFT-SHORT-CPD-63":
        architecture = "TFT"
        total_time_steps = 63
        changepoint_lbws = [63]
    else:
        raise ValueError(f"Invalid experiment: {experiment}")

    versions = range(1, num_repeats + 1) if not TEST_MODE else [1]

    prefix = (
        NAME
        + ("_TEST" if TEST_MODE else "")
        + ("" if TRAIN_VALID_RATIO == 0.90 else f"_split{int(TRAIN_VALID_RATIO * 100)}")
    )

    # make a unique project name per repeat
    cp_string = "none" if not changepoint_lbws else "".join(map(str, changepoint_lbws))
    time_string = "time" if TIME_FEATURES else "notime"
    diversifier = "div" if EVALUATE_DIVERSIFIED_VAL_SHARPE else "val"

    base_name = (
        f"{prefix}_{architecture.lower()}_cp{cp_string}"
        f"_len{total_time_steps}_{time_string}_{diversifier}_v"
    )

    for v in versions:
        PROJECT_NAME = base_name + str(v)

        intervals = [
            (train_start, y, y + test_window_size)
            for y in range(test_start, test_end)
        ]

        # start from your fixed-params grid and then override
        params = MODLE_PARAMS.copy()
        params.update(
            total_time_steps=total_time_steps,
            architecture=architecture,
            evaluate_diversified_val_sharpe=EVALUATE_DIVERSIFIED_VAL_SHARPE,
            train_valid_ratio=TRAIN_VALID_RATIO,
            time_features=TIME_FEATURES,
            force_output_sharpe_length=FORCE_OUTPUT_SHARPE_LENGTH,
        )

        if TEST_MODE:
            params["num_epochs"] = 1
            params["random_search_iterations"] = 2

        # ------------------------------------------------------------
        # ←—— **HERE** is the only block you need to swap out
        # switch from the old `quandl_cpd_...` to your openbb files:
        # ------------------------------------------------------------
        if changepoint_lbws:
            max_lbw = max(changepoint_lbws)
            features_file_path = FEATURES_OPENBB_FILE_PATH(max_lbw)
        else:
            # none → openbb_cpd_none lbw
            features_file_path = FEATURES_OPENBB_FILE_PATH(None)
        # ------------------------------------------------------------

        # run the backtest
        run_all_windows(
            PROJECT_NAME,
            features_file_path,
            intervals,
            params,
            changepoint_lbws,
            ASSET_CLASS_MAPPING,
            [32, 64, 128] if total_time_steps == 252 else HP_MINIBATCH_SIZE,
            test_window_size,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DMN experiment")
    parser.add_argument(
        "experiment",
        metavar="c",
        type=str,
        nargs="?",
        default="TFT-CPD-126-21",
        choices=[
            "LSTM",
            "LSTM-CPD-21",
            "LSTM-CPD-63",
            "TFT",
            "TFT-CPD-126-21",
            "TFT-SHORT",
            "TFT-SHORT-CPD-21",
            "TFT-SHORT-CPD-63",
        ],
        help="Which model + CPD variant to run",
    )
    parser.add_argument(
        "train_start",
        metavar="s",
        type=int,
        nargs="?",
        default=1990,
        help="Training start year",
    )
    parser.add_argument(
        "test_start",
        metavar="t",
        type=int,
        nargs="?",
        default=2016,
        help="Test start year",
    )
    parser.add_argument(
        "test_end",
        metavar="e",
        type=int,
        nargs="?",
        default=2022,
        help="Test end year",
    )
    parser.add_argument(
        "test_window_size",
        metavar="w",
        type=int,
        nargs="?",
        default=1,
        help="Test window size (years)",
    )
    parser.add_argument(
        "num_repeats",
        metavar="r",
        type=int,
        nargs="?",
        default=1,
        help="How many repeats of each backtest",
    )

    args = parser.parse_args()
    main(
        args.experiment,
        args.train_start,
        args.test_start,
        args.test_end,
        args.test_window_size,
        args.num_repeats,
    )
