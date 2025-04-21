import csv
import datetime as dt
from typing import Dict, List, Optional, Tuple, Union
import gc

import gpflow
import numpy as np
import pandas as pd
import tensorflow as tf
from gpflow.kernels import ChangePoints, Matern32
from sklearn.preprocessing import StandardScaler
from tensorflow_probability import bijectors as tfb

from tqdm import tqdm
import os
import multiprocessing
import time
import psutil
import sys

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Suppress INFO messages

# Try to properly configure GPU
try:
    # Prevent TensorFlow from allocating all GPU memory at once
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPU is available: {len(gpus)} GPU(s) detected")
    else:
        print("No GPU found. Running on CPU")
except Exception as e:
    print(f"Error configuring GPU, falling back to CPU: {e}")
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Force CPU usage

Kernel = gpflow.kernels.base.Kernel

MAX_ITERATIONS = 200


class ChangePointsWithBounds(ChangePoints):
    def __init__(
        self,
        kernels: Tuple[Kernel, Kernel],
        location: float,
        interval: Tuple[float, float],
        steepness: float = 1.0,
        name: Optional[str] = None,
    ):
        """Overwrite the Chnagepoints class to
        1) only take a single location
        2) so location is bounded by interval


        Args:
            kernels (Tuple[Kernel, Kernel]): the left hand and right hand kernels
            location (float): changepoint location initialisation, must lie within interval
            interval (Tuple[float, float]): the interval which bounds the changepoint hyperparameter
            steepness (float, optional): initialisation of the steepness parameter. Defaults to 1.0.
            name (Optional[str], optional): class name. Defaults to None.

        Raises:
            ValueError: errors if intial changepoint location is not within interval
        """
        # overwrite the locations variable to enforce bounds
        if location < interval[0] or location > interval[1]:
            raise ValueError(
                "Location {loc} is not in range [{low},{high}]".format(
                    loc=location, low=interval[0], high=interval[1]
                )
            )
        locations = [location]
        super().__init__(
            kernels=kernels,
            locations=locations,
            steepness=steepness,
            name=name,
        )

        affine = tfb.Shift(tf.cast(interval[0], tf.float64))(
            tfb.Scale(tf.cast(interval[1] - interval[0], tf.float64))
        )
        self.locations = gpflow.base.Parameter(
            locations,
            transform=tfb.Chain([affine, tfb.Sigmoid()]),
            dtype=tf.float64,
        )

    def _sigmoids(self, X: tf.Tensor) -> tf.Tensor:
        # overwrite to remove sorting of locations
        locations = tf.reshape(self.locations, (1, 1, -1))
        steepness = tf.reshape(self.steepness, (1, 1, -1))
        return tf.sigmoid(steepness * (X[:, :, None] - locations))


def fit_matern_kernel(
    time_series_data: pd.DataFrame,
    variance: float = 1.0,
    lengthscale: float = 1.0,
    likelihood_variance: float = 1.0,
) -> Tuple[float, Dict[str, float]]:
    """Fit the Matern 3/2 kernel on a time-series

    Args:
        time_series_data (pd.DataFrame): time-series with columns X and Y
        variance (float, optional): variance parameter initialisation. Defaults to 1.0.
        lengthscale (float, optional): lengthscale parameter initialisation. Defaults to 1.0.
        likelihood_variance (float, optional): likelihood variance parameter initialisation. Defaults to 1.0.

    Returns:
        Tuple[float, Dict[str, float]]: negative log marginal likelihood and paramters after fitting the GP
    """
    try:
      m = gpflow.models.GPR(
          data=(
              time_series_data.loc[:, ["X"]].to_numpy(),
              time_series_data.loc[:, ["Y"]].to_numpy(),
          ),
          kernel=Matern32(variance=variance, lengthscales=lengthscale),
          noise_variance=likelihood_variance,
      )
      opt = gpflow.optimizers.Scipy()
      nlml = opt.minimize(
          m.training_loss,
          m.trainable_variables,
          options=dict(maxiter=MAX_ITERATIONS),
      ).fun
      params = {
          "kM_variance": m.kernel.variance.numpy(),
          "kM_lengthscales": m.kernel.lengthscales.numpy(),
          "kM_likelihood_variance": m.likelihood.variance.numpy(),
      }
    except tf.errors.InternalError as e:
      if "CUDA" in str(e):
        print("GPU error detected, retrying with CPU...")
        # Force CPU for this operation
        with tf.device('/CPU:0'):
          m = gpflow.models.GPR(
              data=(
                  time_series_data.loc[:, ["X"]].to_numpy(),
                  time_series_data.loc[:, ["Y"]].to_numpy(),
              ),
              kernel=Matern32(variance=variance, lengthscales=lengthscale),
              noise_variance=likelihood_variance,
          )
          opt = gpflow.optimizers.Scipy()
          nlml = opt.minimize(
              m.training_loss,
              m.trainable_variables,
              options=dict(maxiter=MAX_ITERATIONS),
          ).fun
          params = {
              "kM_variance": m.kernel.variance.numpy(),
              "kM_lengthscales": m.kernel.lengthscales.numpy(),
              "kM_likelihood_variance": m.likelihood.variance.numpy(),
          }
      else:
        raise

    # Clear TF graph and release memory
    del m, opt
    gc.collect()

    return nlml, params


def fit_changepoint_kernel(
    time_series_data: pd.DataFrame,
    k1_variance: float = 1.0,
    k1_lengthscale: float = 1.0,
    k2_variance: float = 1.0,
    k2_lengthscale: float = 1.0,
    kC_likelihood_variance=1.0,
    kC_changepoint_location=None,
    kC_steepness=1.0,
) -> Tuple[float, float, Dict[str, float]]:
    """Fit the Changepoint kernel on a time-series

    Args:
        time_series_data (pd.DataFrame): time-series with ciolumns X and Y
        k1_variance (float, optional): variance parameter initialisation for k1. Defaults to 1.0.
        k1_lengthscale (float, optional): lengthscale initialisation for k1. Defaults to 1.0.
        k2_variance (float, optional): variance parameter initialisation for k2. Defaults to 1.0.
        k2_lengthscale (float, optional): lengthscale initialisation for k2. Defaults to 1.0.
        kC_likelihood_variance (float, optional): likelihood variance parameter initialisation. Defaults to 1.0.
        kC_changepoint_location (float, optional): changepoint location initialisation, if None uses midpoint of interval. Defaults to None.
        kC_steepness (float, optional): steepness parameter initialisation. Defaults to 1.0.

    Returns:
        Tuple[float, float, Dict[str, float]]: changepoint location, negative log marginal likelihood and paramters after fitting the GP
    """
    if not kC_changepoint_location:
        kC_changepoint_location = (
            time_series_data["X"].iloc[0] + time_series_data["X"].iloc[-1]
        ) / 2.0

    m = gpflow.models.GPR(
        data=(
            time_series_data.loc[:, ["X"]].to_numpy(),
            time_series_data.loc[:, ["Y"]].to_numpy(),
        ),
        kernel=ChangePointsWithBounds(
            [
                Matern32(variance=k1_variance, lengthscales=k1_lengthscale),
                Matern32(variance=k2_variance, lengthscales=k2_lengthscale),
            ],
            location=kC_changepoint_location,
            interval=(
                time_series_data["X"].iloc[0],
                time_series_data["X"].iloc[-1],
            ),
            steepness=kC_steepness,
        ),
    )
    m.likelihood.variance.assign(kC_likelihood_variance)
    opt = gpflow.optimizers.Scipy()
    nlml = opt.minimize(
        m.training_loss, m.trainable_variables, options=dict(maxiter=200)
    ).fun

    # Extract parameters and values before cleanup
    changepoint_location = m.kernel.locations[0].numpy()
    params = {
        "k1_variance": m.kernel.kernels[0].variance.numpy().flatten()[0],
        "k1_lengthscale": m.kernel.kernels[0]
        .lengthscales.numpy()
        .flatten()[0],
        "k2_variance": m.kernel.kernels[1].variance.numpy().flatten()[0],
        "k2_lengthscale": m.kernel.kernels[1]
        .lengthscales.numpy()
        .flatten()[0],
        "kC_likelihood_variance": m.likelihood.variance.numpy().flatten()[0],
        "kC_changepoint_location": changepoint_location,
        "kC_steepness": m.kernel.steepness.numpy(),
    }

    # Clear TF graph and release memory
    del m, opt
    gc.collect()

    return changepoint_location, nlml, params


def changepoint_severity(
    kC_nlml: Union[float, List[float]], kM_nlml: Union[float, List[float]]
) -> float:
    """Changepoint score as detailed in https://arxiv.org/pdf/2105.13727.pdf

    Args:
        kC_nlml (Union[float, List[float]]): negative log marginal likelihood of Changepoint kernel
        kM_nlml (Union[float, List[float]]): negative log marginal likelihood of Matern 3/2 kernel

    Returns:
        float: changepoint score
    """
    normalized_nlml = kC_nlml - kM_nlml
    return 1 - 1 / (np.mean(np.exp(-normalized_nlml)) + 1)


def changepoint_loc_and_score(
    time_series_data_window: pd.DataFrame,
    kM_variance: float = 1.0,
    kM_lengthscale: float = 1.0,
    kM_likelihood_variance: float = 1.0,
    k1_variance: float = None,
    k1_lengthscale: float = None,
    k2_variance: float = None,
    k2_lengthscale: float = None,
    kC_likelihood_variance=1.0,  # TODO note this seems to work better by resetting this
    # kC_likelihood_variance=None,
    kC_changepoint_location=None,
    kC_steepness=1.0,
) -> Tuple[float, float, float, Dict[str, float], Dict[str, float]]:
    """For a single time-series window, calcualte changepoint score and location as detailed in https://arxiv.org/pdf/2105.13727.pdf

    Args:
        time_series_data_window (pd.DataFrame): time-series with columns X and Y
        kM_variance (float, optional): variance initialisation for Matern 3/2 kernel. Defaults to 1.0.
        kM_lengthscale (float, optional): lengthscale initialisation for Matern 3/2 kernel. Defaults to 1.0.
        kM_likelihood_variance (float, optional): likelihood variance initialisation for Matern 3/2 kernel. Defaults to 1.0.
        k1_variance (float, optional): variance initialisation for Changepoint kernel k1, if None uses fitted variance parameter from Matern 3/2. Defaults to None.
        k1_lengthscale (float, optional): lengthscale initialisation for Changepoint kernel k1, if None uses fitted lengthscale parameter from Matern 3/2. Defaults to None.
        k2_variance (float, optional): variance initialisation for Changepoint kernel k2, if None uses fitted variance parameter from Matern 3/2. Defaults to None.
        k2_lengthscale (float, optional): lengthscale initialisation for for Changepoint kernel k2, if None uses fitted lengthscale parameter from Matern 3/2. Defaults to None.
        kC_likelihood_variance ([type], optional): likelihood variance initialisation for Changepoint kernel. Defaults to None.
        kC_changepoint_location ([type], optional): changepoint location initialisation for Changepoint, if None uses midpoint of interval. Defaults to None.
        kC_steepness (float, optional): changepoint location initialisation for Changepoint. Defaults to 1.0.

    Returns:
        Tuple[float, float, float, Dict[str, float], Dict[str, float]]: changepoint score, changepoint location,
        changepoint location normalised by interval length to [0,1], Matern 3/2 kernel parameters, Changepoint kernel parameters
    """

    time_series_data = time_series_data_window.copy()
    Y_data = time_series_data[["Y"]].values
    time_series_data[["Y"]] = StandardScaler().fit(Y_data).transform(Y_data)
    # time_series_data.loc[:, "X"] = time_series_data.loc[:, "X"] - time_series_data.loc[time_series_data.index[0], "X"]
    del Y_data  # free up memory

    try:
        (kM_nlml, kM_params) = fit_matern_kernel(
            time_series_data,
            kM_variance,
            kM_lengthscale,
            kM_likelihood_variance,
        )
    except BaseException as ex:
        # do not want to optimise again if the hyperparameters
        # were already initialised as the defaults
        if kM_variance == kM_lengthscale == kM_likelihood_variance == 1.0:
            raise BaseException(
                "Retry with default hyperparameters - already using default parameters."
            ) from ex
        (
            kM_nlml,
            kM_params,
        ) = fit_matern_kernel(time_series_data)

    is_cp_location_default = (
        (not kC_changepoint_location)
        or kC_changepoint_location < time_series_data["X"].iloc[0]
        or kC_changepoint_location > time_series_data["X"].iloc[-1]
    )
    if is_cp_location_default:
        # default to midpoint
        kC_changepoint_location = (
            time_series_data["X"].iloc[-1] + time_series_data["X"].iloc[0]
        ) / 2.0

    if not k1_variance:
        k1_variance = kM_params["kM_variance"]

    if not k1_lengthscale:
        k1_lengthscale = kM_params["kM_lengthscales"]

    if not k2_variance:
        k2_variance = kM_params["kM_variance"]

    if not k2_lengthscale:
        k2_lengthscale = kM_params["kM_lengthscales"]

    if not kC_likelihood_variance:
        kC_likelihood_variance = kM_params["kM_likelihood_variance"]

    try:
        (changepoint_location, kC_nlml, kC_params) = fit_changepoint_kernel(
            time_series_data,
            k1_variance=k1_variance,
            k1_lengthscale=k1_lengthscale,
            k2_variance=k2_variance,
            k2_lengthscale=k2_lengthscale,
            kC_likelihood_variance=kC_likelihood_variance,
            kC_changepoint_location=kC_changepoint_location,
            kC_steepness=kC_steepness,
        )
    except BaseException as ex:
        # do not want to optimise again if the hyperparameters
        # were already initialised as the defaults
        if (
            k1_variance
            == k1_lengthscale
            == k2_variance
            == k2_lengthscale
            == kC_likelihood_variance
            == kC_steepness
            == 1.0
        ) and is_cp_location_default:
            raise BaseException(
                "Retry with default hyperparameters - already using default parameters."
            ) from ex
        (
            changepoint_location,
            kC_nlml,
            kC_params,
        ) = fit_changepoint_kernel(time_series_data)

    cp_score = changepoint_severity(kC_nlml, kM_nlml)
    cp_loc_normalised = (
        time_series_data["X"].iloc[-1] - changepoint_location
    ) / (time_series_data["X"].iloc[-1] - time_series_data["X"].iloc[0])

    # Clean up memory
    gc.collect()

    return (
        cp_score,
        changepoint_location,
        cp_loc_normalised,
        kM_params,
        kC_params,
    )

# Add this function to process a single time window
def process_time_window(args):
    """Process a single time window
    Returns (window_date, time_index, cp_loc, cp_loc_normalised, cp_score)
    """
    ts_data_window, time_index, use_kM_hyp_to_initialise_kC = args
    
    window_date = ts_data_window["date"].iloc[-1].strftime("%Y-%m-%d")
    
    try:
        if use_kM_hyp_to_initialise_kC:
            cp_score, cp_loc, cp_loc_normalised, _, _ = changepoint_loc_and_score(
                ts_data_window,
            )
        else:
            cp_score, cp_loc, cp_loc_normalised, _, _ = changepoint_loc_and_score(
                ts_data_window,
                k1_lengthscale=1.0,
                k1_variance=1.0,
                k2_lengthscale=1.0,
                k2_variance=1.0,
                kC_likelihood_variance=1.0,
            )
    except Exception as e:
        # write as NA when fails and will deal with this later
        cp_score, cp_loc, cp_loc_normalised = "NA", "NA", "NA"
        print(f"Error processing window {window_date}: {str(e)}")
    
    # Clear memory after each window
    gc.collect()
    
    return [window_date, time_index, cp_loc, cp_loc_normalised, cp_score]

# Modify the run_module function to process time windows in batches
def run_module(
    time_series_data: pd.DataFrame,
    lookback_window_length: int,
    output_csv_file_path: str,
    start_date: dt.datetime = None,
    end_date: dt.datetime = None,
    use_kM_hyp_to_initialise_kC=True,
    batch_size: int = 10,  # Process time windows in batches
    memory_threshold: int = 80,  # Memory threshold to pause processing
):
    """Run the changepoint detection module as described in https://arxiv.org/pdf/2105.13727.pdf
    for all times (in date range if specified). Outputs results to a csv.
    
    This version processes time windows in parallel batches for improved memory efficiency.
    """
    # Previous data preparation code remains the same
    if start_date and end_date:
        first_window = time_series_data.loc[:start_date].iloc[
            -(lookback_window_length + 1) :, :
        ]
        remaining_data = time_series_data.loc[start_date:end_date, :]
        if remaining_data.index[0] == start_date:
            remaining_data = remaining_data.iloc[1:, :]
        else:
            first_window = first_window.iloc[1:]
        time_series_data = pd.concat([first_window, remaining_data]).copy()
    elif not start_date and not end_date:
        time_series_data = time_series_data.copy()
    elif not start_date:
        time_series_data = time_series_data.iloc[:end_date, :].copy()
    elif not end_date:
        first_window = time_series_data.loc[:start_date].iloc[
            -(lookback_window_length + 1) :, :
        ]
        remaining_data = time_series_data.loc[start_date:, :]
        if remaining_data.index[0] == start_date:
            remaining_data = remaining_data.iloc[1:, :]
        else:
            first_window = first_window.iloc[1:]
        time_series_data = pd.concat([first_window, remaining_data]).copy()

    # Create output CSV file
    csv_fields = ["date", "t", "cp_location", "cp_location_norm", "cp_score"]
    with open(output_csv_file_path, "w") as f:
        writer = csv.writer(f)
        writer.writerow(csv_fields)
    
    # For progress tracking
    progress_file = output_csv_file_path + ".progress"
    processed_windows = set()
    
    # Check if we're resuming an interrupted run
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            processed_windows = set(int(line.strip()) for line in f.readlines())
        print(f"Resuming from previous run. Already processed {len(processed_windows)} time windows")

    # Prepare time series data
    time_series_data["date"] = time_series_data.index
    time_series_data = time_series_data.reset_index(drop=True)
    
    # Process time windows in batches
    total_windows = len(time_series_data) - lookback_window_length - 1
    
    # Add memory monitoring
    def check_memory_usage():
        """Check memory usage and return percentage"""
        try:
            import psutil
            memory = psutil.virtual_memory()
            return memory.percent
        except ImportError:
            return 0  # If psutil not available, don't monitor memory
    
    # Determine available workers (use fewer workers for each ticker to save memory)
    n_workers = min(multiprocessing.cpu_count() - 1, 4)  # Cap at 4 workers per ticker
    print(f"Processing with {n_workers} workers")
    
    # Process in batches
    for batch_start in tqdm(
        range(lookback_window_length + 1, len(time_series_data), batch_size),
        desc="Processing time windows",
        total=(total_windows + batch_size - 1) // batch_size,
        unit="batch"
    ):
        # Check memory before starting a new batch
        memory_usage = check_memory_usage()
        if memory_usage > memory_threshold:
            print(f"Memory usage high ({memory_usage}%). Pausing to clean up...")
            # Force garbage collection
            gc.collect()
            tf.keras.backend.clear_session()
            # Wait for memory to clear
            time.sleep(5)
            
            # If memory still high, reduce batch size
            if check_memory_usage() > memory_threshold:
                old_batch_size = batch_size
                batch_size = max(1, batch_size // 2)
                print(f"Reducing batch size from {old_batch_size} to {batch_size}")
        
        batch_end = min(batch_start + batch_size, len(time_series_data))
        
        # Prepare window data for this batch
        batch_windows = []
        for window_end in range(batch_start, batch_end):
            # Skip already processed windows
            if window_end - 1 in processed_windows:
                continue
                
            ts_data_window = time_series_data.iloc[
                window_end - (lookback_window_length + 1) : window_end
            ][["date", "daily_returns"]].copy()
            ts_data_window["X"] = ts_data_window.index.astype(float)
            ts_data_window = ts_data_window.rename(columns={"daily_returns": "Y"})
            time_index = window_end - 1
            
            batch_windows.append((ts_data_window, time_index, use_kM_hyp_to_initialise_kC))
        
        # If all windows in this batch have already been processed, skip it
        if not batch_windows:
            continue
            
        # Process this batch in parallel
        with multiprocessing.Pool(processes=n_workers) as process_pool:
            results = list(process_pool.map(process_time_window, batch_windows))
        
        # Write results to CSV
        with open(output_csv_file_path, "a") as f:
            writer = csv.writer(f)
            for result in results:
                writer.writerow(result)
                # Mark this window as processed
                processed_windows.add(result[1])
        
        # Update progress file
        with open(progress_file, "w") as f:
            for window_idx in processed_windows:
                f.write(f"{window_idx}\n")
        
        # Clear memory after each batch
        del batch_windows
        gc.collect()
        tf.keras.backend.clear_session()

