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
import tensorflow as tf
from scipy import stats
from sklearn.model_selection import StratifiedKFold
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


# class DataCleanerX(_DataCleaner):
#     def __init__(self, df: pd.DataFrame) -> None:
#         super().__init__(df)
#         self._dropna()
#
#     def _dropna(self):
#         self._df.dropna()
#
#     def _remove_outliers(self) -> None:
#         pass
#
#     def _remove_duplicates(self) -> None:
#         pass


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

    # @staticmethod
    # def _enrich_with_future_timepoints(df: pd.DataFrame,
    #                                    n_future_time_points: int = 8,
    #                                    timepoints_resolution_in_minutes: int = 15) -> pd.DataFrame:
    #     """
    #     Extracting the m next time points (difference from time zero)
    #     :param n_future_time_points: number of future time points
    #     :return:
    #     """
    #     for i, g in enumerate(range(timepoints_resolution_in_minutes,
    #                                 timepoints_resolution_in_minutes * (n_future_time_points + 1),
    #                                 timepoints_resolution_in_minutes),
    #                           1):
    #         new_header = f'{settings.DataStructureGlucose.GLUCOSE_VALUE_HEADER} difference +%0.1dmin' % g
    #         df[new_header] = \
    #             df[settings.DataStructureGlucose.GLUCOSE_VALUE_HEADER].shift(-i) \
    #             - df[settings.DataStructureGlucose.GLUCOSE_VALUE_HEADER]
    #     return df.dropna(how='any', axis=0).drop(settings.DataStructureGlucose.GLUCOSE_VALUE_HEADER, axis=1)


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
        self._df.dropna(inplace=True)

    # @staticmethod
    # def create_shifts(df: pd.DataFrame,
    #                   feature_name: str,
    #                   n_previous_time_points: int = 48) -> pd.DataFrame:
    #     """
    #     Creating a data frame with columns corresponding to previous time points
    #     :param feature_name:
    #     :param df: A pandas data frame
    #     :param n_previous_time_points: number of previous time points to shift
    #     :return:
    #     """
    #     for i, g in enumerate(
    #             range(settings.DataStructureGlucose.SAMPLING_INTERVAL_IN_MINUTES,
    #                   settings.DataStructureGlucose.SAMPLING_INTERVAL_IN_MINUTES * (n_previous_time_points + 1),
    #                   settings.DataStructureGlucose.SAMPLING_INTERVAL_IN_MINUTES),
    #             1):
    #         df[f'{feature_name} -%0.1dmin' % g] = df[f'{feature_name}'].shift(i)
    #     return df.dropna(how='any', axis=0)

    # def _filter_X_by_glucose_indices(self,
    #                                  glucose_value_header: str = 'GlucoseValue') -> None:
    #     print('[Dataset] _filter_X_by_glucose_indices')
    #     self._df = self._df.dropna(subset=[glucose_value_header]).sort_index()
    #


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

    def build_X_raw_from_glucose_and_meals_datasets(self,
                                                    glucose_dataset: Dataset,
                                                    meals_dataset: Dataset) -> None:
        glucose_df = glucose_dataset.get_processed()
        meals_df = meals_dataset.get_processed()
        self._raw = pd.concat([glucose_df, meals_df], axis=1, join='outer').sort_index()

    # def get_X_and_y(self,
    #                 n_previous_time_points: int = 48,
    #                 n_future_time_points: int = 8) \
    #         -> Tuple[pd.DataFrame, pd.DataFrame]:
    #     """
    #     Returns X and y as dataframes. Contains only timepoints with enough past and future samples.
    #     """
    #     X = self._processed[settings.DataStructureGlucose.GLUCOSE_VALUE_HEADER].reset_index() \
    #         .groupby(settings.DataStructure.ID_HEADER) \
    #         .apply(DataProcessorX.create_shifts,
    #                feature_name=settings.DataStructureGlucose.GLUCOSE_VALUE_HEADER,
    #                n_previous_time_points=n_previous_time_points) \
    #         .set_index([settings.DataStructure.ID_HEADER, settings.DataStructure.DATE_HEADER])
    #     y = self._processed[settings.DataStructureGlucose.GLUCOSE_VALUE_HEADER].reset_index() \
    #         .groupby(settings.DataStructure.ID_HEADER) \
    #         .apply(DataEnricherX._enrich_with_future_timepoints,
    #                n_future_time_points=n_future_time_points) \
    #         .set_index([settings.DataStructure.ID_HEADER, settings.DataStructure.DATE_HEADER])
    #     idx_intersection = X.index.intersection(y.index)
    #     return X.loc[idx_intersection], y.loc[idx_intersection]

    def get_multivariate_X_and_y(self,
                                 num_of_past_timepoints: int = 48,
                                 num_of_future_timepoints: int = 8) \
            -> Sequence[Tuple[np.ndarray, np.ndarray]]:
        """
        Returns a 2-tuple of 3D np.ndarray with the following shapes:
        X's shape is (# of instances in the dataset, # past timepoints + 1, # of features),
        y's shape is (# of future timepoints, 1).
        """
        # X, y = self.get_X_and_y(n_previous_time_points=n_previous_time_points,
        #                         n_future_time_points=n_future_time_points)
        # multivariate_y = y.values.reshape(y.shape + (1,))
        #
        # shifted_Xs = []
        # for feature_name in X.columns:
        #     shifted_X = X[feature_name].reset_index().groupby('id') \
        #         .apply(DataProcessorX.create_shifts,
        #                feature_name=feature_name,
        #                n_previous_time_points=48) \
        #         .set_index(['id', 'Date'])
        #     shifted_Xs.append(shifted_X)
        # multivariate_X = np.dstack(shifted_Xs)
        # return multivariate_X, multivariate_y

        shifted_X = self.get_processed()
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
        multivariate_X = multivariate_X[num_of_past_timepoints:, :, :]
        shifted_X = self.get_processed()[settings.DataStructureGlucose.GLUCOSE_VALUE_HEADER]
        for i in range(num_of_future_timepoints):
            shifted_X = shifted_X.groupby(level=0).shift(-1)
            multivariate_y[:, i] = shifted_X
        multivariate_y = multivariate_y[:-num_of_future_timepoints]
        return multivariate_X, multivariate_y


class Predictor:
    def __init__(self):
        self._nn = None

    def load(self, checkpoint_dir_name: str, checkpoint_num: Optional[int] = None) -> None:
        checkpoint_dir = os.path.join(settings.Files.CHECKPOINTS_DIR_NAME, checkpoint_dir_name)
        checkpoint_path = os.path.join(checkpoint_dir, 'cp-{epoch:04d}.ckpt')
        if checkpoint_num is None:
            latest = tf.train.latest_checkpoint(checkpoint_dir)
            self._nn.load_weights(latest)
        else:
            self._nn = tf.keras.models.load_model(checkpoint_path.format(checkpoint_num))

    def save(self, checkpoint_dir_name: str) -> None:
        self._nn.save(checkpoint_dir_name)

    def reset(self) -> None:
        self._nn = settings.NN.MODEL.compile(optimizer=settings.NN.OPTIMIZER, loss=settings.NN.LOSS)


class Trainer:
    def __init__(self, predictor: Predictor):
        self._predictor = predictor

    def train(self, dataset: DatasetX, num_of_epochs: int) -> tf.keras.callbacks.History:
        checkpoint_dir_name = Trainer._generate_new_checkpoint_dir_name()
        checkpoint_dir = os.path.join(settings.Files.CHECKPOINTS_DIR_NAME, checkpoint_dir_name)
        print(f'starting training... checkpoints are saved at: {checkpoint_dir}')
        checkpoint_path = os.path.join(checkpoint_dir, 'cp-{epoch:04d}.ckpt')
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True,
                                                         verbose=1)
        multivariate_X, multivariate_y = dataset.get_multivariate_X_and_y()
        for train_idx, valid_idx in StratifiedKFold(settings.TrainingConfiguration.CROSS_VALIDATION_NUM_OF_FOLDS):
            train_X, train_y = multivariate_X[train_idx], multivariate_y[train_idx]
            valid_X, valid_y = multivariate_X[valid_idx], multivariate_y[valid_idx]
            self._predictor.fit(train_X,
                                train_y,
                                epochs=num_of_epochs,
                                steps_per_epoch=settings.TrainingConfiguration.EVALUATION_INTERVAL,
                                validation_data=(valid_X, valid_y),
                                validation_steps=settings.TrainingConfiguration.VALIDATION_STEPS,
                                callbacks=[cp_callback],
                                verbose=1)

    @staticmethod
    def _generate_new_checkpoint_dir_name():
        return '{date:%Y-%m-%d_%H_%M_%S}'.format(date=datetime.datetime.now())

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


# class Evaluator
#
# def compute_mean_pearson(y_true, y_pred, individual_index_name='id', n_future_time_points=8):
#     """
#     This function takes the true glucose values and the predicted ones, flattens the data per individual and then
#     computed the Pearson correlation between the two vectors per individual.
#
#     **This is how we will evaluate your predictions, you may use this function in your code**
#
#     :param y_true: an M by n_future_time_points data frame holding the true glucose values
#     :param y_pred: an M by n_future_time_points data frame holding the predicted glucose values
#     :param individual_index_name: the name of the individual's indeces, default is 'id'
#     :param n_future_time_points: number of future time points to predict, default is 8
#     :return: the mean Pearson correlation
#     """
#     # making sure y_true and y_pred are of the same size
#     assert y_true.shape == y_pred.shape
#     # making sure y_true and y_pred share the same exact indeces and index names
#     assert (y_true.index == y_pred.index).all() and y_true.index.names == y_pred.index.names
#     # making sure that individual_index_name is a part of the index of both dataframes
#     assert individual_index_name in y_true.index.names and individual_index_name in y_pred.index.names
#
#     # concat data frames
#     joined_df = pd.concat((y_true, y_pred), axis=1)
#     return joined_df.groupby(individual_index_name) \
#         .apply(lambda x: pearsonr(x.iloc[:, :n_future_time_points].values.ravel(),
#                                   x.iloc[:, n_future_time_points:].values.ravel())[0]).mean()
