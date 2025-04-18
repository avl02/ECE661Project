import argparse
from typing import List

import pandas as pd

from data.pull_data import pull_openbb_sample_data
from settings.default import (
    OPENBB_2003_TICKERS,
    CPD_OPENBB_OUTPUT_FOLDER,
    FEATURES_OPENBB_FILE_PATH,
)
from mom_trans.data_prep import (
    deep_momentum_strategy_features,
    include_changepoint_features,
)


def main(
    tickers: List[str],
    cpd_module_folder: str,
    lookback_window_length: int,
    output_file_path: str,
    extra_lbw: List[int],
):
    # Generate deep momentum features for each ticker
    features = pd.concat(
        [
            deep_momentum_strategy_features(
                pull_openbb_sample_data(ticker)
            ).assign(ticker=ticker)
            for ticker in tickers
        ]
    )

    # Prepare DataFrame index and column for saving
    features.date = features.index
    features.index.name = "Date"

    if lookback_window_length:
        # Include primary CPD features
        features_w_cpd = include_changepoint_features(
            features, cpd_module_folder, lookback_window_length
        )

        # Optionally merge additional CPD windows
        for extra in extra_lbw:
            extra_path = output_file_path.replace(
                f"openbb_cpd_{lookback_window_length}lbw.csv",
                f"openbb_cpd_{extra}lbw.csv",
            )
            extra_data = pd.read_csv(
                extra_path, index_col=0, parse_dates=True
            ).reset_index()[["date", "ticker", f"cp_rl_{extra}", f"cp_score_{extra}"]]
            extra_data["date"] = pd.to_datetime(extra_data["date"])

            # Merge on (date, ticker)
            features_w_cpd = pd.merge(
                features_w_cpd.set_index(["date", "ticker"]),
                extra_data.set_index(["date", "ticker"]),
                left_index=True,
                right_index=True,
            ).reset_index()
            features_w_cpd.index = features_w_cpd["date"]
            features_w_cpd.index.name = "Date"

        # Save with CPD columns
        features_w_cpd.to_csv(output_file_path)
    else:
        # Save base features only
        features.to_csv(output_file_path)


if __name__ == "__main__":

    def get_args():
        parser = argparse.ArgumentParser(
            description="Generate deep momentum features with optional CPD for OpenBB data"
        )
        parser.add_argument(
            "lookback_window_length",
            metavar="l",
            type=int,
            nargs="?",
            default=None,
            help="Primary CPD lookback window length (days)",
        )
        parser.add_argument(
            "extra_lbw",
            metavar="-e",
            type=int,
            nargs="*",
            default=[],
            help="Additional CPD lookback windows (days)",
        )

        args = parser.parse_known_args()[0]

        return (
            OPENBB_2003_TICKERS,
            CPD_OPENBB_OUTPUT_FOLDER(args.lookback_window_length),
            args.lookback_window_length,
            FEATURES_OPENBB_FILE_PATH(args.lookback_window_length),
            args.extra_lbw,
        )

    main(*get_args())
