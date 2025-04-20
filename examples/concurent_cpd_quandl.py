import multiprocessing
import argparse
import os
import time
from tqdm import tqdm

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress INFO messages

from settings.default import (
    QUANDL_TICKERS,
    CPD_QUANDL_OUTPUT_FOLDER,
    CPD_DEFAULT_LBW,
    OPENBB_2003_TICKERS,
    CPD_OPENBB_OUTPUT_FOLDER,
)

N_WORKERS = min(32, len(OPENBB_2003_TICKERS), multiprocessing.cpu_count()-1)
print(f"Running with {N_WORKERS} workers")

# Process tickers in batches to prevent memory exhaustion
BATCH_SIZE = 18

def main(lookback_window_length: int, batch_size: int = BATCH_SIZE):
    if not os.path.exists(CPD_OPENBB_OUTPUT_FOLDER(lookback_window_length)):
        os.mkdir(CPD_OPENBB_OUTPUT_FOLDER(lookback_window_length))

    # Process tickers in batches
    total_batches = (len(OPENBB_2003_TICKERS) + batch_size - 1) // batch_size
    for batch_idx in tqdm(range(total_batches), desc="Processing batches", unit = "batch"):
    # for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(OPENBB_2003_TICKERS))
        batch_tickers = OPENBB_2003_TICKERS[start_idx:end_idx]
        
        print(f"Processing batch {batch_idx + 1}/{total_batches} with {len(batch_tickers)} tickers")
        
        batch_processes = [
            f'python -m examples.cpd_quandl "{ticker}" "{os.path.join(CPD_OPENBB_OUTPUT_FOLDER(lookback_window_length), ticker + ".csv")}" "1990-01-01" "2021-12-31" "{lookback_window_length}" "10"'
            for ticker in batch_tickers
        ]
        
        # Track progress of individual ticker processing within the batch
        with multiprocessing.Pool(processes=N_WORKERS) as process_pool:
          list(tqdm(
            process_pool.imap(os.system, batch_processes),
            total = len(batch_processes),
            desc= f"Batch {batch_idx + 1} progress",
            unit = "ticker"
          ))
        # process_pool = multiprocessing.Pool(processes=N_WORKERS)
        # process_pool.map(os.system, batch_processes)
        # process_pool.close()
        # process_pool.join()
        
        print(f"Completed batch {batch_idx + 1}/{total_batches}")
        time.sleep(0.5)
        

    # all_processes = [
    #     f'python -m examples.cpd_quandl "{ticker}" "{os.path.join(CPD_OPENBB_OUTPUT_FOLDER(lookback_window_length), ticker + ".csv")}" "1990-01-01" "2021-12-31" "{lookback_window_length}"'
    #     for ticker in OPENBB_2003_TICKERS
    # ]
    # process_pool = multiprocessing.Pool(processes=N_WORKERS)
    # process_pool.map(os.system, all_processes)


if __name__ == "__main__":

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
        # return [
        #     parser.parse_known_args()[0].lookback_window_length,
        # ]
        return[
          args.lookback_window_length,
          args.batch_size,
        ]

    main(*get_args())
