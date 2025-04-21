import multiprocessing
import argparse
import os
import time
import psutil
import signal
import sys
from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress INFO messages

from settings.default import (
    QUANDL_TICKERS,
    CPD_QUANDL_OUTPUT_FOLDER,
    CPD_DEFAULT_LBW,
    OPENBB_2003_TICKERS,
    CPD_OPENBB_OUTPUT_FOLDER,
)

# Configure memory thresholds
MEMORY_WARNING = 75  # Start reducing batch size
MEMORY_CRITICAL = 90  # Pause processing until memory clears
MEMORY_EMERGENCY = 95  # Emergency save and exit


# Process tickers in batches to prevent memory exhaustion
BATCH_SIZE = 7  # Reduced from 18 for better memory management
N_WORKERS = min(
    BATCH_SIZE, len(OPENBB_2003_TICKERS), multiprocessing.cpu_count() - 1
)
print(f"Running with {N_WORKERS} workers")


def check_memory_usage():
    """Check current memory usage and return percentage used"""
    memory = psutil.virtual_memory()
    return memory.percent


def process_ticker(args):
    """
    Process a single ticker with memory monitoring
    Returns (ticker, success_status)
    """
    ticker, command, lookback_window_length = args

    # Check memory before processing each ticker
    memory_usage = check_memory_usage()
    if memory_usage > MEMORY_CRITICAL:
        print(
            f"CRITICAL: Memory usage at {memory_usage}%. Skipping ticker {ticker} for now."
        )
        return ticker, False

    # Check if this ticker has a progress file indicating partial completion
    output_file = os.path.join(CPD_OPENBB_OUTPUT_FOLDER(lookback_window_length), ticker + ".csv")
    progress_file = output_file + ".progress"

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


def main(lookback_window_length: int, batch_size: int = BATCH_SIZE):
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
        t for t in OPENBB_2003_TICKERS if t not in processed_tickers
    ]
    print(f"Remaining tickers to process: {len(remaining_tickers)}")

    if not remaining_tickers:
        print("All tickers have been processed. Nothing to do.")
        return

    # Process tickers in batches
    current_batch_size = batch_size
    total_batches = (
        len(remaining_tickers) + current_batch_size - 1
    ) // current_batch_size
    batch_idx = 0

    # Keep track of newly processed tickers in this run
    newly_processed = set()

    while batch_idx < total_batches:
        # Check memory BEFORE starting a new batch
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

        if memory_usage > MEMORY_WARNING:
            # Reduce batch size when memory gets high
            old_batch_size = current_batch_size
            current_batch_size = max(1, current_batch_size - 2)

            if old_batch_size != current_batch_size:
                print(
                    f"Warning: Memory usage at {memory_usage}%. Reducing batch size: {old_batch_size} → {current_batch_size}"
                )
                # Recalculate total batches with new batch size
                total_batches = (
                    len(remaining_tickers)
                    - batch_idx * old_batch_size
                    + current_batch_size
                    - 1
                ) // current_batch_size
        elif (
            memory_usage < (MEMORY_WARNING - 20)
            and current_batch_size < batch_size
        ):
            # If memory usage is comfortably low, increase batch size slightly
            current_batch_size = min(batch_size, current_batch_size + 2)
            print(
                f"Memory usage is low ({memory_usage}%). Increasing batch size to {current_batch_size}"
            )

        # Calculate indices for current batch
        start_idx = sum(
            batch_sizes for batch_sizes in [current_batch_size] * batch_idx
        )
        end_idx = min(start_idx + current_batch_size, len(remaining_tickers))
        batch_tickers = remaining_tickers[start_idx:end_idx]

        if not batch_tickers:  # Safety check
            break

        print(
            f"Processing batch {batch_idx + 1}/{total_batches} with {len(batch_tickers)} tickers"
        )

        # Create commands for this batch
        batch_commands = [
            (
                ticker,
                f'python -m examples.cpd_quandl "{ticker}" "{os.path.join(CPD_OPENBB_OUTPUT_FOLDER(lookback_window_length), ticker + ".csv")}" "1990-01-01" "2021-12-31" "{lookback_window_length}" "10"',
                lookback_window_length,
            )
            for ticker in batch_tickers
        ]

        # Process tickers with memory-aware wrapper
        with multiprocessing.Pool(processes=N_WORKERS) as process_pool:
            results = list(
                tqdm(
                    process_pool.imap(process_ticker, batch_commands),
                    total=len(batch_commands),
                    desc=f"Batch {batch_idx + 1} progress",
                    unit="ticker",
                )
            )

        # Track successfully processed tickers
        successful_tickers = [ticker for ticker, success in results if success]
        print(
            f"Completed batch {batch_idx + 1}: {len(successful_tickers)}/{len(batch_tickers)} successful"
        )

        # Update progress file AFTER each batch
        with open(progress_file, "a") as f:
            for ticker in successful_tickers:
                f.write(f"{ticker}\n")
                newly_processed.add(ticker)

        # Give the system a break between batches
        time.sleep(1)
        batch_idx += 1


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
            "batch_size",
            metavar="b",
            type=int,
            nargs="?",
            default=BATCH_SIZE,
            help="Number of tickers to process in each batch",
        )
        args = parser.parse_known_args()[0]
        return [
            args.lookback_window_length,
            args.batch_size,
        ]

    main(*get_args())
