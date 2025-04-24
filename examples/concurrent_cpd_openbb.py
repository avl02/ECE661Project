import argparse
import os
import time
import psutil
import signal
import sys
from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress INFO messages

from settings.default import (
    CPD_DEFAULT_LBW,
    OPENBB_2003_TICKERS,
    CPD_OPENBB_OUTPUT_FOLDER,
)

# Configure memory thresholds
MEMORY_WARNING = 75  # Start reducing batch size
MEMORY_CRITICAL = 90  # Pause processing until memory clears
MEMORY_EMERGENCY = 95  # Emergency save and exit

# Process tickers one at a time, but with parallel time window processing
TIME_WINDOW_BATCH_SIZE = 10  # Number of time windows to process in parallel


def check_memory_usage():
    """Check current memory usage and return percentage used"""
    memory = psutil.virtual_memory()
    return memory.percent


def process_ticker(ticker, lookback_window_length):
    """Process a single ticker sequentially"""
    command = f'python -m examples.cpd_openbb "{ticker}" "{os.path.join(CPD_OPENBB_OUTPUT_FOLDER(lookback_window_length), ticker + ".csv")}" "1990-01-01" "2021-12-31" "{lookback_window_length}" "{TIME_WINDOW_BATCH_SIZE}"'

    try:
        result = os.system(command)
        return ticker, result == 0
    except Exception as e:
        print(f"Error processing ticker {ticker}: {str(e)}")
        return ticker, False


def wait_for_memory_to_clear(target_usage=MEMORY_WARNING - 10):
    """Pause execution until memory usage drops below target"""
    while check_memory_usage() > target_usage:
        print(
            f"Waiting for memory to clear... Current: {check_memory_usage()}%"
        )
        time.sleep(10)  # Wait 10 seconds before checking again


def emergency_exit(processed_tickers, progress_file):
    """Save progress and exit gracefully when memory is critically high"""
    print("EMERGENCY: Memory usage critical. Saving progress and exiting...")

    # Save all successfully processed tickers
    with open(progress_file, "a") as f:
        for ticker in processed_tickers:
            f.write(f"{ticker}\n")

    print(
        f"Progress saved to {progress_file}. Run the script again to resume."
    )
    sys.exit(1)


def main(
    lookback_window_length: int,
    time_window_batch_size: int = TIME_WINDOW_BATCH_SIZE,
    tickers=None,
):
    global TIME_WINDOW_BATCH_SIZE
    TIME_WINDOW_BATCH_SIZE = time_window_batch_size

    # Use provided tickers or default to all tickers
    target_tickers = tickers if tickers else OPENBB_2003_TICKERS

    if not os.path.exists(CPD_OPENBB_OUTPUT_FOLDER(lookback_window_length)):
        os.makedirs(
            CPD_OPENBB_OUTPUT_FOLDER(lookback_window_length), exist_ok=True
        )

    # Setup progress tracking
    progress_file = os.path.join(
        CPD_OPENBB_OUTPUT_FOLDER(lookback_window_length),
        "processed_tickers.txt",
    )
    processed_tickers = set()

    # Load already processed tickers if file exists
    if os.path.exists(progress_file):
        with open(progress_file, "r") as f:
            processed_tickers = set(line.strip() for line in f.readlines())
        print(
            f"Resuming from previous run. Already processed {len(processed_tickers)} tickers"
        )

    # Filter out already processed tickers
    remaining_tickers = [
        t for t in target_tickers if t not in processed_tickers
    ]
    print(f"Remaining tickers to process: {len(remaining_tickers)}")

    if not remaining_tickers:
        print("All tickers have been processed. Nothing to do.")
        return

    # Process tickers one at a time
    newly_processed = set()

    for ticker_idx, ticker in enumerate(
        tqdm(remaining_tickers, desc="Processing tickers")
    ):
        # Check memory before starting a new ticker
        memory_usage = check_memory_usage()

        if memory_usage > MEMORY_EMERGENCY:
            # Emergency save and exit
            emergency_exit(newly_processed, progress_file)

        if memory_usage > MEMORY_CRITICAL:
            # Critical memory usage - pause and wait for memory to clear
            print(
                f"CRITICAL: Memory usage at {memory_usage}%. Pausing execution..."
            )
            wait_for_memory_to_clear(MEMORY_WARNING - 5)
            continue  # Re-check memory after waiting

        print(
            f"Processing ticker {ticker_idx + 1}/{len(remaining_tickers)}: {ticker}"
        )

        # Process this ticker
        ticker_result = process_ticker(ticker, lookback_window_length)

        # Track successfully processed ticker
        if ticker_result[1]:
            print(f"Successfully processed ticker {ticker}")
            with open(progress_file, "a") as f:
                f.write(f"{ticker}\n")
            newly_processed.add(ticker)
        else:
            print(f"Failed to process ticker {ticker}")

        # Give the system a break between tickers
        time.sleep(1)


if __name__ == "__main__":
    # Register signal handler for graceful termination
    def signal_handler(sig, frame):
        print("You pressed Ctrl+C! Exiting gracefully...")
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)

    def get_args():
        """Returns settings from command line."""
        parser = argparse.ArgumentParser(
            description="Run changepoint detection module for all tickers"
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
            "time_window_batch_size",
            metavar="b",
            type=int,
            nargs="?",
            default=TIME_WINDOW_BATCH_SIZE,
            help="Number of time windows to process in parallel",
        )
        parser.add_argument(
            "--tickers",
            "-t",
            type=str,
            nargs="+",
            default=None,
            help="List of ticker symbols to process (default: all tickers)",
        )
        args = parser.parse_known_args()[0]
        return [
            args.lookback_window_length,
            args.time_window_batch_size,
            args.tickers,
        ]

    main(*get_args())
