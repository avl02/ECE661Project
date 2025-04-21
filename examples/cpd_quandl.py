import argparse
import datetime as dt
import os

import pandas as pd

import mom_trans.changepoint_detection as cpd
from mom_trans.data_prep import calc_returns
from data.pull_data import pull_openbb_sample_data

from settings.default import CPD_DEFAULT_LBW, USE_KM_HYP_TO_INITIALISE_KC


def main(
    ticker: str,
    output_file_path: str,
    start_date: dt.datetime,
    end_date: dt.datetime,
    lookback_window_length: int,
    batch_size: int = 10, # Add batch_size parameter with default value
):

    print(f"Processing ticker {ticker} from {start_date} to {end_date}")
    print(f"LBW={lookback_window_length}, batch size={batch_size}")

    try:
      # Check if we already have a partially completed CSV
      progress_file = output_file_path+".progress"
      is_resuming = os.path.exists(progress_file)

      if is_resuming:
          print(f"Found progress file for {ticker}, resuming from previous run")

      # Read data
      data = pull_openbb_sample_data(ticker)
      data["daily_returns"] = calc_returns(data["close"])

      result = cpd.run_module(
        data,
        lookback_window_length,
        output_file_path,
        start_date,
        end_date,
        USE_KM_HYP_TO_INITIALISE_KC,
        batch_size=batch_size,
      )

      return result # Return success status from run_module

    except Exception as e:
      print(f"Error processing ticker {ticker}: {str(e)}")
      return False


    # cpd.run_module(
    #     data,
    #     lookback_window_length,
    #     output_file_path,
    #     start_date,
    #     end_date,
    #     USE_KM_HYP_TO_INITIALISE_KC,
    #     batch_size=batch_size, # Pass batch_size to control memory usage
    # )


if __name__ == "__main__":

    def get_args():
        """Returns settings from command line."""

        parser = argparse.ArgumentParser(
            description="Run changepoint detection module"
        )
        parser.add_argument(
            "ticker",
            metavar="t",
            type=str,
            nargs="?",
            default="ICE_SB",
            # choices=[],
            help="Ticker type",
        )
        parser.add_argument(
            "output_file_path",
            metavar="f",
            type=str,
            nargs="?",
            default="data/test.csv",
            # choices=[],
            help="Output file location for csv.",
        )
        parser.add_argument(
            "start_date",
            metavar="s",
            type=str,
            nargs="?",
            default="1990-01-01",
            help="Start date in format yyyy-mm-dd",
        )
        parser.add_argument(
            "end_date",
            metavar="e",
            type=str,
            nargs="?",
            default="2021-12-31",
            help="End date in format yyyy-mm-dd",
        )
        parser.add_argument(
            "lookback_window_length",
            metavar="l",
            type=int,
            nargs="?",
            default=CPD_DEFAULT_LBW,
            help="CPD lookback window length",
        )

        parser.add_argument(
            "batch_size",
            metavar="b",
            type=int,
            nargs="?",
            default=10,
            help="Batch size for processing windows (helps manage memory usage)",
        )

        args = parser.parse_known_args()[0]

        start_date = dt.datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = dt.datetime.strptime(args.end_date, "%Y-%m-%d")

        return (
            args.ticker,
            args.output_file_path,
            start_date,
            end_date,
            args.lookback_window_length,
            args.batch_size
        )

    main(*get_args())
