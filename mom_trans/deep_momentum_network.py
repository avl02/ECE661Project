import os, json, shutil, copy, collections
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import keras_tuner as kt
from keras_tuner.tuners.randomsearch import RandomSearch
from empyrical import sharpe_ratio

from settings.hp_grid import (
    HP_HIDDEN_LAYER_SIZE,
    HP_DROPOUT_RATE,
    HP_MAX_GRADIENT_NORM,
    HP_LEARNING_RATE,
    HP_MINIBATCH_SIZE,
)
from mom_trans.model_inputs import ModelFeatures


# ───────────────────────────── custom loss (maximises Sharpe) ────────────────
class SharpeLoss(tf.keras.losses.Loss):
    def __init__(self, output_size: int = 1):
        super().__init__()
        self.output_size = output_size

    def call(self, y_true, weights):
        captured = weights * y_true
        μ = tf.reduce_mean(captured)
        σ = tf.sqrt(tf.reduce_mean(tf.square(captured)) - tf.square(μ) + 1e-9)
        return -(μ / σ) * tf.sqrt(252.0)        # annualised Sharpe


# ─────────────────────────── validation callback (Sharpe) ────────────────────
class SharpeValidationLoss(keras.callbacks.Callback):
    def __init__(
        self,
        inputs, returns, time_idx, n_time,
        patience, n_workers,
        save_to="tmp/best.ckpt",
        min_delta=1e-4,
    ):
        super().__init__()
        self.x        = inputs
        self.r        = returns
        self.time_idx = time_idx
        self.n_time   = n_time
        self.patience = patience
        self.n_workers = n_workers
        self.save_to  = save_to
        self.min_delta = min_delta
        self.best, self.wait = -np.inf, 0

    # keras‑tuner will update this path for every trial
    def set_weights_save_loc(self, loc):
        self.save_to = loc

    def on_epoch_end(self, epoch, logs=None):
        pos = self.model.predict(
            self.x, use_multiprocessing=True, workers=self.n_workers
        )
        ret = tf.cast(self.r, pos.dtype)
        captured = tf.math.unsorted_segment_mean(
            pos * ret, self.time_idx, self.n_time
        )[1:]                                   # drop dummy idx 0
        sharpe = (
            tf.reduce_mean(captured)
            / tf.sqrt(tf.math.reduce_variance(captured) + 1e-9)
            * tf.sqrt(252.0)
        ).numpy()
        logs["sharpe"] = sharpe

        if sharpe > self.best + self.min_delta:
            # ── make sure folder exists then save ──────────────────────
            os.makedirs(os.path.dirname(self.save_to), exist_ok=True)
            self.model.save_weights(self.save_to)
            self.best, self.wait = sharpe, 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.model.stop_training = True
                # ── only load if the checkpoint is really there ────────
                if os.path.exists(self.save_to):
                    self.model.load_weights(self.save_to)


# ───────────────────────── tuner mix‑in that injects batch_size ──────────────
class _BatchSizeTuner(RandomSearch):
    def __init__(self, hp_minibatch_size, *a, **kw):
        self.hp_minibatch_size = hp_minibatch_size
        super().__init__(*a, **kw)

    def run_trial(self, trial, *args, **kwargs):
        kwargs["batch_size"] = trial.hyperparameters.Choice(
            "batch_size", self.hp_minibatch_size
        )
        super().run_trial(trial, *args, **kwargs)

    # Keras‑Tuner deep‑copies callbacks; keep ours intact
    def _deepcopy_callbacks(self, callbacks):
        return list(callbacks) if callbacks else []


class TunerValidationLoss(_BatchSizeTuner):
    pass


class TunerDiversifiedSharpe(_BatchSizeTuner):
    def __init__(self, executions_per_trial=1, *a, **kw):
        self.executions_per_trial = executions_per_trial
        super().__init__(*a, **kw)

    def run_trial(self, trial, *args, **kwargs):
        kwargs["batch_size"] = trial.hyperparameters.Choice(
            "batch_size", self.hp_minibatch_size
        )
        orig_cbs = kwargs.pop("callbacks", [])
        for cb in orig_cbs:
            if isinstance(cb, SharpeValidationLoss):
                cb.set_weights_save_loc(
                    self._get_checkpoint_fname(trial.trial_id, self._reported_step)
                )

        metrics = collections.defaultdict(list)
        for _ in range(self.executions_per_trial):
            fit_kw = copy.copy(kwargs)
            cbs = list(orig_cbs)
            cbs.append(kt.engine.tuner_utils.TunerCallback(self, trial))
            fit_kw["callbacks"] = cbs
            hist = self._build_and_fit_model(trial, args, fit_kw)
            for m, vals in hist.history.items():
                best = np.min(vals) if self.oracle.objective.direction == "min" else np.max(vals)
                metrics[m].append(best)

        self.oracle.update_trial(
            trial.trial_id,
            metrics={m: float(np.mean(v)) for m, v in metrics.items()},
            step=self._reported_step,
        )


# ─────────────────────────────── base DMN class ──────────────────────────────
class DeepMomentumNetworkModel(ABC):
    # ---------------------------------------------------------------------
    def __init__(self, project_name, hp_dir, hp_minibatch_size, **params):
        p = params.copy()
        self.time_steps  = int(p.pop("total_time_steps"))
        self.input_size  = int(p.pop("input_size"))
        self.output_size = int(p.pop("output_size"))
        self.n_workers   = int(p.pop("multiprocessing_workers"))
        self.num_epochs  = int(p.pop("num_epochs"))
        self.patience    = int(p.pop("early_stopping_patience"))
        self.max_trials  = int(p.pop("random_search_iterations"))
        self.diversified = bool(p.pop("evaluate_diversified_val_sharpe"))
        self.force_out   = p.pop("force_output_sharpe_length", None)

        print("Deep Momentum Network params:")
        for k, v in p.items(): print(f"  {k} = {v!r}")

        def build(hp): return self.model_builder(hp)

        tuner_kw = dict(
            hypermodel   = build,
            max_trials   = self.max_trials,
            directory    = hp_dir,
            project_name = project_name,
        )
        if self.diversified:
            self.tuner = TunerDiversifiedSharpe(
                objective=kt.Objective("sharpe", "max"),
                hp_minibatch_size=hp_minibatch_size,
                **tuner_kw,
            )
        else:
            self.tuner = TunerValidationLoss(
                objective="val_loss",
                hp_minibatch_size=hp_minibatch_size,
                **tuner_kw,
            )

    # helper --------------------------------------------------------------
    @staticmethod
    def _index_times(t):
        uniq = np.sort(np.unique(t))
        if uniq[0] != "": uniq = np.insert(uniq, 0, "")
        mapping = dict(zip(uniq, range(len(uniq))))
        return np.vectorize(mapping.__getitem__)(t), len(mapping)

    # ---------------------------------------------------------------------
    def hyperparameter_search(self, train, valid):
        X, y, w, _, _       = ModelFeatures._unpack(train)
        Xv, yv, wv, _, tv   = ModelFeatures._unpack(valid)

        if self.diversified:
            idx, n = self._index_times(tv)
            cbs = [
                SharpeValidationLoss(Xv, yv, idx, n, self.patience, self.n_workers),
                tf.keras.callbacks.TerminateOnNaN(),
            ]
            self.tuner.search(
                x=X, y=y, sample_weight=w,
                epochs=self.num_epochs,
                callbacks=cbs,
                shuffle=True,
                use_multiprocessing=True, workers=self.n_workers,
            )
        else:
            cbs=[keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=self.patience, restore_best_weights=True
            )]
            self.tuner.search(
                x=X, y=y, sample_weight=w,
                epochs=self.num_epochs,
                validation_data=(Xv, yv, wv),
                callbacks=cbs,
                shuffle=True,
                use_multiprocessing=True, workers=self.n_workers,
            )

        hp_best   = self.tuner.get_best_hyperparameters(1)[0].values
        model_best= self.tuner.get_best_models(1)[0]
        return hp_best, model_best

    # ---------------------------------------------------------------------
    def load_model(self, hp_vals):
        hp = kt.HyperParameters()
        hp.values = hp_vals
        return self.tuner.hypermodel.build(hp)

    # ---------------------------------------------------------------------
    def fit(self, train, valid, hp_vals, tmp_ckpt="tmp/best.ckpt"):
        X, y, w, _, _       = ModelFeatures._unpack(train)
        Xv, yv, wv, _, tv   = ModelFeatures._unpack(valid)
        model = self.load_model(hp_vals)

        if self.diversified:
            idx, n = self._index_times(tv)
            cbs = [SharpeValidationLoss(
                        Xv, yv, idx, n,
                        self.patience, self.n_workers,
                        save_to=tmp_ckpt
                   ),
                   tf.keras.callbacks.TerminateOnNaN()]
            model.fit(
                X, y, sample_weight=w,
                epochs=self.num_epochs,
                batch_size=hp_vals["batch_size"],
                callbacks=cbs,
                shuffle=True,
                use_multiprocessing=True, workers=self.n_workers,
            )
            if os.path.exists(tmp_ckpt):
                model.load_weights(tmp_ckpt)
        else:
            cbs=[keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=self.patience,
                    restore_best_weights=True)]
            model.fit(
                X, y, sample_weight=w,
                epochs=self.num_epochs,
                batch_size=hp_vals["batch_size"],
                validation_data=(Xv, yv, wv),
                callbacks=cbs,
                shuffle=True,
                use_multiprocessing=True, workers=self.n_workers,
            )
        return model

    # ---------------------------------------------------------------------
    def evaluate(self, data, model):
        X, y, w, _, _ = ModelFeatures._unpack(data)
        if self.diversified:
            _, perf = self.get_positions(data, model, False)
            return perf
        else:
            vals = model.evaluate(
                X, y, sample_weight=w,
                use_multiprocessing=True, workers=self.n_workers,
            )
            return dict(zip(model.metrics_names, vals))["loss"]

    # ---------------------------------------------------------------------
    def get_positions(
        self, data, model,
        sliding_window=True,
        years_geq=np.iinfo(np.int32).min,
        years_lt =np.iinfo(np.int32).max,
    ):
        X, y, _, ident, time = ModelFeatures._unpack(data)
        if sliding_window:
            time = pd.to_datetime(time[:, -1, 0].flatten())
            years= time.year
            ident= ident[:, -1, 0].flatten()
            rets = y[:, -1, 0].flatten()
        else:
            time = pd.to_datetime(time.flatten())
            years= time.year
            ident= ident.flatten()
            rets = y.flatten()

        mask = (years >= years_geq) & (years < years_lt)

        pos = model.predict(X, use_multiprocessing=True, workers=self.n_workers)
        pos = pos[:, -1, 0].flatten() if sliding_window else pos.flatten()

        cap = rets * pos
        df = pd.DataFrame(dict(
            identifier=ident[mask],
            time=time[mask],
            returns=rets[mask],
            position=pos[mask],
            captured_returns=cap[mask]))
        sharpe = sharpe_ratio(df.groupby("time")["captured_returns"].sum())
        return df, sharpe

    # ---------------------------------------------------------------------
    @abstractmethod
    def model_builder(self, hp): ...


# ────────────────────────────────── concrete LSTM ────────────────────────────
class LstmDeepMomentumNetworkModel(DeepMomentumNetworkModel):
    def __init__(self, project_name, hp_dir, hp_minibatch_size=HP_MINIBATCH_SIZE, **params):
        super().__init__(project_name, hp_dir, hp_minibatch_size, **params)

    def model_builder(self, hp):
        hidden   = hp.Choice("hidden_layer_size",  HP_HIDDEN_LAYER_SIZE)
        dropout  = hp.Choice("dropout_rate",       HP_DROPOUT_RATE)
        clipnorm = hp.Choice("max_gradient_norm",  HP_MAX_GRADIENT_NORM)
        lr       = hp.Choice("learning_rate",      HP_LEARNING_RATE)

        inp = keras.Input((self.time_steps, self.input_size))
        x   = keras.layers.LSTM(hidden, return_sequences=True, dropout=dropout)(inp)
        x   = keras.layers.Dropout(dropout)(x)
        out = keras.layers.TimeDistributed(
                keras.layers.Dense(self.output_size, activation="tanh",
                                   kernel_constraint=keras.constraints.max_norm(3))
              )(x)

        model = keras.Model(inp, out)
        model.compile(
            optimizer = keras.optimizers.legacy.Adam(learning_rate=lr, clipnorm=clipnorm),
            loss      = SharpeLoss(self.output_size).call,
            sample_weight_mode="temporal",
        )
        return model
