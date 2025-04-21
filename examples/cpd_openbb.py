import argparse
import datetime as dt
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress INFO messages

from mom_trans.data_prep import calc_returns
from data.pull_data import pull_openbb_sample_data
import mom_trans.cp_detection as cpd

# Add memory management imports
import gc
import psutil
import tensorflow as tf


def check_memory_usage():
    """Check current memory usage and return percentage used"""
    memory = psutil.virtual_memory()
    return memory.percent


def main(
    ticker: str,
    output_csv_file_path: str,
    start_date: dt.datetime,
    end_date: dt.datetime,
    lookback_window_length: int,
    batch_size: int = 7,  # Process multiple time windows at once
    memory_threshold: int = 85,  # Memory threshold to pause processing
):
    """Run the changepoint detection module on a single ticker
    but parallel-process time windows in batches
    """

    print(f"Processing ticker {ticker} from {start_date} to {end_date}")
    print(f"LBW={lookback_window_length}, batch size={batch_size}")

    try:
        # Read data
        data = pull_openbb_sample_data(ticker)
        data["daily_returns"] = calc_returns(data["close"])

        # Implement batch processing of time windows within run_module
        cpd.run_module(
            data,
            lookback_window_length,
            output_csv_file_path,
            start_date=start_date,
            end_date=end_date,
            batch_size=batch_size,
            memory_threshold=memory_threshold,
        )

        # Clear memory after processing
        del data
        gc.collect()
        tf.keras.backend.clear_session()

        return True

    except Exception as e:
        print(f"Error processing ticker {ticker}: {str(e)}")
        return False


if __name__ == "__main__":

    def get_args():
        """Returns settings from command line."""
        parser = argparse.ArgumentParser(
            description="Run changepoint detection module for a quandl ticker"
        )
        parser.add_argument(
            "ticker_name", metavar="t", type=str, help="ticker symbol"
        )
        parser.add_argument(
            "output_csv_file_path",
            metavar="o",
            type=str,
            help="output csv file path",
        )
        parser.add_argument(
            "start_date_str", metavar="s", type=str, help="start date"
        )
        parser.add_argument(
            "end_date_str", metavar="e", type=str, help="end date"
        )
        parser.add_argument(
            "lookback_window_length",
            metavar="l",
            type=int,
            help="lookback window length",
        )
        parser.add_argument(
            "time_window_batch_size",
            metavar="b",
            type=int,
            nargs="?",
            default=10,
            help="number of time windows to process in parallel",
        )
        args = parser.parse_known_args()[0]
        return [
            args.ticker_name,
            args.output_csv_file_path,
            args.start_date_str,
            args.end_date_str,
            args.lookback_window_length,
            args.time_window_batch_size,
        ]

    main(*get_args())
