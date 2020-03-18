####################################################################################################
# wis_dnn_challenge.py
# Description: This is a template file for the WIS DNN challenge submission.
# Important: The only thing you should not change is the signature of the class (Predictor) and its predict function.
#            Anything else is for you to decide how to implement.
#            We provide you with a very basic working version of this class.
#
# Authors: Lior_Baltiansky Nadav_Timor
#
# Python 3.6
####################################################################################################

import settings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import tensorflow as tf
from scipy import stats
from sklearn.model_selection import KFold
from typing import List, Tuple, Sequence, Type, Optional
import os
import datetime


class _DataCleaner:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df
        self._clean()

    def get_cleaned(self) -> pd.DataFrame:
        return self._df

    def _clean(self) -> None:
        self._remove_outliers()
        self._remove_duplicates()

    def _remove_outliers(self) -> None:
        pass

    def _remove_duplicates(self) -> None:
        pass

    # @classmethod
    # def _remove_ids_with_missing_data(self, ids_with_data: pd.Index) -> None:
    #     ids_with_data_mask = self._df.index.isin(ids_with_data, level=0)
    #     self._df = self._df[ids_with_data_mask]


class DataCleanerGlucose(_DataCleaner):
    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__(df)
        self._zero_center()

    def _remove_outliers(self) -> None:
        self._df = self._df.rolling(5, center=True, min_periods=1).median()

    def _remove_duplicates(self) -> None:
        indices = [settings.DataStructure.ID_HEADER, settings.DataStructure.DATE_HEADER]
        self._df = self._df.groupby(indices).median()

    def _zero_center(self) -> None:
        self._df = self._df.groupby(settings.DataStructure.ID_HEADER).transform(lambda x: x - x.mean())


class DataCleanerMeals(_DataCleaner):
    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__(df)

    def _remove_outliers(self) -> None:
        keep_mask = np.abs(stats.zscore(self._df[settings.DataStructureMeals.FEATURES_WITH_OUTLIERS])) < 3
        self._df = self._df[keep_mask]

    def _remove_duplicates(self) -> None:
        indices = [settings.DataStructure.ID_HEADER, settings.DataStructure.DATE_HEADER]
        self._df = self._df.groupby(indices).sum()


class _DataEnricher:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df
        self._enrich()

    def get_enriched(self) -> pd.DataFrame:
        return self._df

    def _enrich(self) -> None:
        pass


class DataEnricherX(_DataEnricher):
    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__(df)
        self._enrich_with_rolling_window_mean(settings.DataStructureGlucose.GLUCOSE_CORRELATED_FEATURES)
        self._enrich_with_cyclic_time()

    def _enrich_with_rolling_window_mean(self,
                                         headers: List[str],
                                         win_size: int = 8,
                                         win_type: str = 'exponential') -> None:
        df_filled = self._df.fillna(0)
        # windowing
        new_headers = [f'{h}_{win_type}_{win_size}' for h in headers]
        if win_type == 'triang':
            win = df_filled[headers].rolling(win_size, win_type=win_type)
        elif win_type == 'exponential':
            win = df_filled[headers].ewm(span=win_size)
        else:
            raise NotImplementedError
        new_df = win.mean()
        # enrich original glucose df
        new_df.columns = new_headers
        self._df = self._df[[settings.DataStructureGlucose.GLUCOSE_VALUE_HEADER]].join(new_df, how='outer')

    def _enrich_with_cyclic_time(self) -> None:
        """
        Encode the time in the day in a cyclic way (23:55 and 00:05 are 10 minutes apart,
                                                    rather than 23 hours and 50 minutes apart).
        Reference: https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/
        """
        MINUTES_IN_A_DAY = 60 * 24
        time = self._df.index.get_level_values('Date').to_series().apply(lambda d: d.time())
        f = 2 * np.pi * time.apply(lambda t: t.hour * 60 + t.minute)
        self._df['sin_time'] = (np.sin(f) / MINUTES_IN_A_DAY).values
        self._df['cos_time'] = (np.cos(f) / MINUTES_IN_A_DAY).values


class _DataProcessor:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def get_processed(self) -> pd.DataFrame:
        return self._df.copy()

    def _clean(self, data_cleaner: Type[_DataCleaner]) -> None:
        data_cleaner = data_cleaner(self._df)
        self._df = data_cleaner.get_cleaned()

    def _enrich(self, data_enricher: Type[_DataEnricher]) -> None:
        data_enricher = data_enricher(self._df)
        self._df = data_enricher.get_enriched()

    # def _normalize(self):
    #     print('[Dataset] _normalize')
    #     #         return (df - df.mean()) / (df.max() - df.min())
    #     return (df - df.mean()) / df.std()


class DataProcessorGlucose(_DataProcessor):
    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__(df)
        self._clean(DataCleanerGlucose)


class DataProcessorMeals(_DataProcessor):
    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__(df)
        self._clean(DataCleanerMeals)


class DataProcessorX(_DataProcessor):
    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__(df)
        self._enrich(DataEnricherX)
        self._dropna()

    def _dropna(self) -> None:
        self._df = self._df.dropna()


class Dataset:
    def __init__(self):
        self._raw = None
        self._processed = None

    def load_raw(self,
                 filename: str,
                 dir_path: str = '',
                 index_col: List[int] = settings.DataStructure.INDEX_COLUMNS,
                 parse_dates: List[str] = settings.DataStructure.HEADERS_WITH_DATES) -> None:
        """
        Load a dataframe from the given file, and set it to be the raw dataset.
        """
        print('[Dataset] load_raw')
        path = os.path.join(dir_path, filename)
        print(f'loading the file: {path}')
        self._raw = pd.read_csv(path, index_col=index_col, parse_dates=parse_dates)

    def set_raw(self, df: pd.DataFrame) -> None:
        """
        Set the given datafeame to be the raw dataset.
        """
        print('[Dataset] set_raw')
        self._raw = df.copy()

    def get_raw(self) -> pd.DataFrame:
        """
        Returns the raw dataset.
        """
        print('[Dataset] get_raw')
        return self._raw.copy()

    def load_processed(self,
                       filename: str,
                       dir_path: str = '',
                       index_col: List[int] = settings.DataStructure.INDEX_COLUMNS,
                       parse_dates: List[str] = settings.DataStructure.HEADERS_WITH_DATES) -> None:
        """
        Load a dataframe from the given file, and set it to be the processed dataset.
        """
        print('[Dataset] load_processed')
        path = os.path.join(dir_path, filename)
        print(f'loading the file: {path}')
        self._processed = pd.read_csv(path, index_col=index_col, parse_dates=parse_dates)

    def set_processed(self, df: pd.DataFrame) -> None:
        """
        Set the given df to be the processed dataset.
        """
        print('[Dataset] set_processed')
        self._processed = df.copy()

    def get_processed(self) -> pd.DataFrame:
        """
        Returns the processed dataset.
        """
        print('[Dataset] get_processed')
        return self._processed.copy()

    def get_processed_shape(self) -> Sequence[Tuple[int, int]]:
        return self._processed.shape

    def save_processed(self, filename: str, dir_path: str = '') -> None:
        print('[Dataset] save_processed')
        path = os.path.join(dir_path, filename)
        print(f'saving processed dataset to file: {path}')
        self._processed.reset_index().to_csv(path, index=False)

    def process(self, data_processor: Type[_DataProcessor]) -> None:
        """
        Process the raw dataset, and set the result as the processed dataset.
        """
        print('[Dataset] process')
        print(f'process data using {data_processor}')
        data_processor = data_processor(self._raw)
        self._processed = data_processor.get_processed()


class DatasetX(Dataset):
    def __init__(self):
        super().__init__()

    def build_raw_X_from_glucose_and_meals_datasets(self,
                                                    glucose_dataset: Dataset,
                                                    meals_dataset: Dataset) -> None:
        glucose_df = glucose_dataset.get_processed()
        meals_df = meals_dataset.get_processed()
        self._raw = pd.concat([glucose_df, meals_df], axis=1, join='outer').sort_index()

    def get_multivariate_X_and_y(self,
                                 num_of_past_timepoints: int = 48,
                                 num_of_future_timepoints: int = 8) \
            -> Sequence[Tuple[np.ndarray, np.ndarray]]:
        """
        Returns a 2-tuple of 3D np.ndarray with the following shapes:
        X's shape is (# of instances in the dataset, # past timepoints + 1, # of features),
        y's shape is (# of future timepoints, 1).
        """
        shifted_X = self.get_processed()
        shifted_X_copy = shifted_X.copy()
        num_of_instances = shifted_X.shape[0]
        num_of_features = shifted_X.shape[1]

        multivariate_X = np.zeros((num_of_instances,
                                   num_of_past_timepoints + 1,
                                   num_of_features))
        multivariate_y = np.zeros((num_of_instances,
                                   num_of_future_timepoints))
        multivariate_X[:, 0, :] = shifted_X.groupby(level=0).shift(0)
        for i in range(1, num_of_past_timepoints + 1):
            shifted_X = shifted_X.groupby(level=0).shift(1)
            multivariate_X[:, i, :] = shifted_X
        indices_to_keep = ~np.isnan(multivariate_X).any(axis=(1, 2))
        shifted_X = shifted_X_copy[settings.DataStructureGlucose.GLUCOSE_VALUE_HEADER] # self.get_processed()[settings.DataStructureGlucose.GLUCOSE_VALUE_HEADER]
        for i in range(num_of_future_timepoints):
            shifted_X = shifted_X.groupby(level=0).shift(-1)
            multivariate_y[:, i] = shifted_X
        indices_to_keep = np.logical_and(indices_to_keep, ~np.isnan(multivariate_y).any(axis=1))
        return multivariate_X[indices_to_keep], multivariate_y[indices_to_keep]


class Predictor:
    def __init__(self):
        self._strategy = tf.distribute.MirroredStrategy()
        self.nn = None
        self.reset()

    def predict(self, X_glucose: pd.DataFrame, X_meals: pd.DataFrame) -> pd.DataFrame:
        self.load(settings.NN.BEST.LOGS_DIR_NAME, settings.NN.BEST.CHECKPOINT_NUM)
        glucose_dataset = Dataset()
        glucose_dataset.set_raw(X_glucose)
        glucose_dataset.process(DataProcessorGlucose)
        meals_dataset = Dataset()
        meals_dataset.set_raw(X_meals)
        meals_dataset.process(DataProcessorMeals)
        dataset = DatasetX()
        dataset.build_raw_X_from_glucose_and_meals_datasets(glucose_dataset=glucose_dataset,
                                                            meals_dataset=meals_dataset)
        dataset.process(DataProcessorX)
        multivariate_X, _ = dataset.get_multivariate_X_and_y()
        self.load(settings.NN.BEST.LOGS_DIR_NAME, settings.NN.BEST.CHECKPOINT_NUM)
        return self.nn.predict(multivariate_X)

    def load(self, logs_dir_name: str, checkpoint_num: Optional[int] = None) -> None:
        self._load_model(logs_dir_name)
        self._load_weights(logs_dir_name=logs_dir_name,
                           checkpoint_num=checkpoint_num)

    def _load_model(self, logs_dir_name: str):
        with self._strategy.scope():
            self.nn = tf.keras.models.load_model(os.path.join(settings.Files.LOGS_DIR_NAME,
                                                          logs_dir_name,
                                                          settings.Files.SAVED_MODEL_DIR_NAME))

    def _load_weights(self, logs_dir_name: str, checkpoint_num: Optional[int] = None) -> None:
        checkpoint_dir = os.path.join(settings.Files.LOGS_DIR_NAME,
                                      logs_dir_name,
                                      settings.Files.CHECKPOINTS_DIR_NAME)
        if checkpoint_num is None:
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)
        else:
            checkpoint_path = os.path.join(checkpoint_dir, f'cp-{checkpoint_num:04d}.ckpt')
        print(f'load model weights from: {checkpoint_path}')
        with self._strategy.scope():
            self.nn.load_weights(checkpoint_path)

    # def save(self, checkpoint_dir_name: str) -> None:
    #     self.nn.save(checkpoint_dir_name)

    def reset(self) -> None:
        with self._strategy.scope():
            self.nn = settings.NN.get_model()
            self.nn.compile(optimizer=settings.NN.OPTIMIZER, loss=settings.NN.LOSS)


class Fitter:
    def __init__(self, predictor: Predictor, dataset: DatasetX, num_of_epochs: int) -> None:
        self._predictor = predictor
        self._dataset = dataset
        self._num_of_epochs = num_of_epochs
        self._logs_dir_name = Fitter._generate_a_unique_logs_dir_name()
        self._saved_model_dir_path = os.path.join(self._logs_dir_name,
                                                  settings.Files.SAVED_MODEL_DIR_NAME)
        self._checkpoints_dir_path = os.path.join(self._logs_dir_name,
                                                  settings.Files.CHECKPOINTS_DIR_NAME)
        self._fit_history_dir_path = os.path.join(self._logs_dir_name,
                                                  settings.Files.FIT_HISTORY_DIR_NAME)
        self._save_model()
        self.fit()

    def _save_model(self):
        self._predictor.nn.save(self._saved_model_dir_path)

    def fit(self) -> List[tf.keras.callbacks.History]:
        print(self._get_message_about_paths())
        tensorboard_callback = self._get_tensorboard_callback()
        cp_callback = self._get_checkpoint_callback()
        multivariate_X, multivariate_y = self._dataset.get_multivariate_X_and_y()
        kf = Fitter._get_kfold()
        history_of_all_folds = []
        for train_idx, valid_idx in kf.split(X=multivariate_X):
            train_X, train_y = multivariate_X[train_idx], multivariate_y[train_idx]
            valid_X, valid_y = multivariate_X[valid_idx], multivariate_y[valid_idx]
            train_dataset = tf.data.Dataset.from_tensor_slices((train_X, train_y))
            train_dataset = train_dataset.cache() \
                                         .shuffle(settings.TrainingConfiguration.BUFFER_SIZE,
                                                  reshuffle_each_iteration=True) \
                                         .batch(settings.TrainingConfiguration.BATCH_SIZE) \
                                         .repeat()
            valid_dataset = tf.data.Dataset.from_tensor_slices((valid_X, valid_y)) \
                                           .batch(settings.TrainingConfiguration.BATCH_SIZE) \
                                           .repeat()
            history = self._predictor.nn.fit(train_dataset,
                                             epochs=self._num_of_epochs,
                                             validation_data=valid_dataset,
                                             verbose=1,
                                             callbacks=[tensorboard_callback, cp_callback],
                                             steps_per_epoch=settings.TrainingConfiguration.STEPS_PER_EPOCH,
                                             validation_steps=settings.TrainingConfiguration.VALIDATION_STEPS)
            history_of_all_folds.append(history)

    @staticmethod
    def _generate_a_unique_logs_dir_name() -> str:
        unique_name_by_datetime = '{date:%Y-%m-%d_%H_%M_%S}'.format(date=datetime.datetime.now())
        return os.path.join(settings.Files.LOGS_DIR_NAME, unique_name_by_datetime)

    def _get_message_about_paths(self) -> str:
        return f'starting training...\n' \
               f'\tcheckpoints will be saved in: {self._checkpoints_dir_path}\n' \
               f'\tfit history logs (can be examined using tensorboard) will be saved in: {self._fit_history_dir_path}'

    @staticmethod
    def _get_kfold() -> KFold:
        return KFold(n_splits=settings.TrainingConfiguration.CROSS_VALIDATION_NUM_OF_FOLDS, shuffle=True)

    def _get_checkpoint_callback(self) -> tf.keras.callbacks.ModelCheckpoint:
        checkpoint_path = os.path.join(self._checkpoints_dir_path, 'cp-{epoch:04d}.ckpt')
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)
        return cp_callback

    def _get_tensorboard_callback(self) -> tf.keras.callbacks.TensorBoard:
        return tf.keras.callbacks.TensorBoard(log_dir=self._fit_history_dir_path, histogram_freq=1)


class Trainer:
    def __init__(self, predictor: Predictor):
        self._predictor = predictor

    def train(self, dataset: DatasetX, num_of_epochs: int) -> List[tf.keras.callbacks.History]:
        fitter = Fitter(predictor=self._predictor, dataset=dataset, num_of_epochs=num_of_epochs)
        history_of_all_folds = fitter.fit()
        return history_of_all_folds

    @staticmethod
    def plot_history(history: tf.keras.callbacks.History) -> None:
        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(len(loss))
        plt.figure()
        plt.plot(epochs, loss, 'b', label='Training loss')
        plt.plot(epochs, val_loss, 'r', label='Validation loss')
        plt.title('Loss history')
        plt.legend()
        plt.show()

# class _Loss:
#     def __init__(self, y_true: pd.DataFrame, y_pred: pd.DataFrame):
#         self._y_true = y_true
#         self._y_pred = y_pred
#
#     def get(self):
#         pass
#
#
# class LossPearson(_Loss):
#     def __init__(self, y_true: pd.DataFrame, y_pred: pd.DataFrame) -> None:
#         super().__init__(y_true, y_pred)
#
#     def get(self) -> float:
#         # making sure y_true and y_pred are of the same size
#         assert self._y_true.shape == self._y_pred.shape
#         # making sure y_true and y_pred share the same exact indeces and index names
#         assert (self._y_true.index == self._y_pred.index).all() and self._y_true.index.names == self._y_pred.index.names
#         # making sure that individual_index_name is a part of the index of both dataframes
#         assert settings.DataStructure.ID_HEADER in self._y_true.index.names \
#                and settings.DataStructure.ID_HEADER in self._y_pred.index.names
#
#         # concat data frames
#         joined_df = pd.concat((self._y_true, self._y_pred), axis=1)
#         return joined_df.groupby(settings.DataStructure.ID_HEADER)\
#                         .apply(lambda x: stats.pearsonr(x.iloc[:, :settings.Challenge.NUM_OF_FUTURE_TIMEPOINTS]\
#                                                          .values\
#                                                          .ravel(),
#                                                         x.iloc[:, settings.Challenge.NUM_OF_FUTURE_TIMEPOINTS:]\
#                                                          .values\
#                                                          .ravel()
#                                                         )[0])\
#                         .mean()


# class _Plotter:
#     def __init__(self, data) -> None:
#         self._data = data
#         self._plot = self._get_plot()
#
#     def _get_plot(self) -> Figure:
#         pass


# class Evaluator:
#     def __init__(self, predictor: Predictor) -> None:
#         self._predictor = predictor
#
#     def get_loss(self, loss: Type[_Loss]):
#         pass
