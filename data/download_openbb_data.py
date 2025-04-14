# # Example of downloading continuous futures data with OpenBB

from openbb import obb
import pandas as pd
from datetime import datetime, timedelta
import os
import time
import argparse
from tqdm import tqdm  # For progress tracking

# Add these imports
import random


def clean_openbb_futures_data(data):
    """
    Clean and process OpenBB futures data into a proper DataFrame

    Parameters:
    -----------
    data : OBBObject
        The data returned from OpenBB's futures.historical or derivatives.futures.historical

    Returns:
    --------
    pd.DataFrame
        DataFrame with date index and 'Settle' column containing closing prices
    """
    try:
        # Check if data is already a DataFrame
        if isinstance(data, pd.DataFrame):
            # Data is already a DataFrame, just extract close prices
            if "close" in data.columns:
                return data[["close"]].rename(columns={"close": "Settle"})
            return data

        # Handle OBBObject with results attribute
        if hasattr(data, "results"):
            # Check if results is a list of tuples (newer OpenBB format)
            if isinstance(data.results[0], tuple) or isinstance(
                data.results[0], list
            ):
                # Create structured data from tuple pairs
                clean_data = []
                for row in data.results:
                    row_dict = {}
                    for field in row:
                        # Each field is a tuple of (field_name, field_value)
                        field_name, field_value = field
                        row_dict[field_name] = field_value
                    clean_data.append(row_dict)

                # Create DataFrame from the list of dictionaries
                df = pd.DataFrame(clean_data)

                # Set date as the index
                if "date" in df.columns:
                    df.set_index("date", inplace=True)

                # Extract close prices and rename
                if "close" in df.columns:
                    return df[["close"]].rename(columns={"close": "Settle"})
                return df

            # Handle case where results is a list of objects with attributes
            elif hasattr(data.results[0], "close"):
                clean_data = []
                for item in data.results:
                    clean_data.append(
                        {
                            "date": getattr(item, "date", None),
                            "close": getattr(item, "close", None),
                        }
                    )
                df = pd.DataFrame(clean_data)
                df.set_index("date", inplace=True)
                return df.rename(columns={"close": "Settle"})

        # If nothing worked, raise an exception
        raise ValueError("Unsupported data format")

    except Exception as e:
        print(f"Error cleaning data: {e}")
        # Return empty DataFrame as fallback
        return pd.DataFrame(columns=["Settle"])


def download_historical_futures(
    symbol, start_date, end_date=None, retry_count=3, retry_delay=2
):
    """
    Download historical futures data with retry logic

    Parameters:
    -----------
    symbol : str
        The futures symbol code (e.g., "CL" for Crude Oil)
    start_date : str or datetime
        Start date for the data
    end_date : str or datetime, optional
        End date for the data, defaults to today
    retry_count : int, optional
        Number of times to retry if API request fails
    retry_delay : int, optional
        Delay between retries in seconds (with jitter)

    Returns:
    --------
    pd.DataFrame or None
        DataFrame with futures data or None if download failed
    """
    if end_date is None:
        end_date = datetime.now()

    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, "%Y-%m-%d")

    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, "%Y-%m-%d")

    # Format dates for API call
    start_str = start_date.strftime("%Y-%m-%d")
    end_str = end_date.strftime("%Y-%m-%d")

    print(f"Downloading {symbol} futures from {start_str} to {end_str}")

    for attempt in range(retry_count):
        try:
            data = obb.derivatives.futures.historical(
                symbol=symbol,
                start_date=start_str,
                end_date=end_str,
                continuous=True,
            )

            if data is not None:
                # Clean and process the data
                df = clean_openbb_futures_data(data)
                return df

        except Exception as e:
            # Check if error message indicates rate limiting
            if any(
                term in str(e).lower()
                for term in ["rate", "limit", "too many", "429"]
            ):
                # Calculate delay with jitter to avoid synchronized retries
                delay = retry_delay + random.uniform(1, 5)
                print(
                    f"Rate limit hit for {symbol}. Waiting {delay:.2f}s before retry {attempt + 1}/{retry_count}"
                )
                time.sleep(delay)
            else:
                print(f"Error downloading {symbol}: {e}")
                if attempt < retry_count - 1:
                    time.sleep(retry_delay)

    print(f"Failed to download {symbol} after {retry_count} attempts")
    return None


def download_all_futures(
    start_date="2000-01-01",
    end_date=None,
    chunk_size=5,
    delay_between_symbols=2,
):
    """
    Download all futures data with rate limit handling

    Parameters:
    -----------
    start_date : str
        Start date for historical data
    end_date : str, optional
        End date for historical data
    chunk_size : int, optional
        Number of futures to process before taking a longer pause
    delay_between_symbols : int, optional
        Delay in seconds between individual symbol downloads
    """
    if not os.path.exists(os.path.join("data", "openbb")):
        os.makedirs(os.path.join("data", "openbb"))

    successful = 0
    failed = 0

    # Get all futures symbols
    tickers = pd.read_csv("data/continuous.csv")
    tickers = tickers["Ticker"].unique()
    OPENBB_SYMBOLS = tickers.tolist()

    # Process in chunks to avoid prolonged API pressure
    for i, symbol in enumerate(tqdm(OPENBB_SYMBOLS)):
        # Add delay between symbols to avoid hitting rate limits
        if i > 0:
            time.sleep(delay_between_symbols)

        # Take a longer pause after each chunk
        if i > 0 and i % chunk_size == 0:
            longer_delay = 15 + random.uniform(0, 5)  # 15-20 second pause
            print(
                f"\nTaking a longer pause of {longer_delay:.1f}s after processing {chunk_size} symbols..."
            )
            time.sleep(longer_delay)

        df = download_historical_futures(symbol, start_date, end_date)

        if df is not None and not df.empty:
            # Save to CSV
            output_file = os.path.join("data", "openbb", f"{symbol}.csv")
            df.to_csv(output_file)
            print(f"Data saved to {output_file} ({len(df)} rows)")
            successful += 1
        else:
            failed += 1

    print(
        f"\nDownload complete. Successfully downloaded {successful} symbols. Failed: {failed}."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Download futures data from OpenBB."
    )
    parser.add_argument(
        "--start", default="2000-01-01", help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument("--end", default=None, help="End date (YYYY-MM-DD)")
    parser.add_argument(
        "--symbol", default=None, help="Single symbol to download (e.g., 'CL')"
    )
    parser.add_argument(
        "--chunk", type=int, default=5, help="Chunk size for batch processing"
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=2,
        help="Delay between symbols (seconds)",
    )

    args = parser.parse_args()

    # Install futures extension if not already installed
    try:
        # Test if futures extension is loaded
        obb.derivatives.futures
    except AttributeError:
        print("Installing futures extension...")
        try:
            obb.extension.install("futures")
            print("Futures extension installed successfully.")
        except Exception as e:
            print(f"Failed to install futures extension: {e}")
            return

    if args.symbol:
        # Download a single symbol
        df = download_historical_futures(args.symbol, args.start, args.end)
        if df is not None:
            output_file = os.path.join("data", "openbb", f"{args.symbol}.csv")
            if not os.path.exists(os.path.join("data", "openbb")):
                os.makedirs(os.path.join("data", "openbb"))
            df.to_csv(output_file)
            print(f"Data saved to {output_file}")
    else:
        # Download all symbols
        download_all_futures(args.start, args.end, args.chunk, args.delay)


if __name__ == "__main__":
    main()
