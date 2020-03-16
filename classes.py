####################################################################################################
# wis_dnn_challenge.py
# Description: This is a template file for the WIS DNN challenge submission.
# Important: The only thing you should not change is the signature of the class (Predictor) and its predict function.
#            Anything else is for you to decide how to implement.
#            We provide you with a very basic working version of this class.
#
# Author: <first name1>_<last name1> [<first name1>_<last name2>]
#
# Python 3.7
####################################################################################################

import pandas as pd
import numpy as np
from scipy import stats
# import torch
import tensorflow as tf
# from tensorflow import keras
import matplotlib.pyplot as plt
import os
import sys
from scipy.stats import pearsonr
from sklearn.model_selection import StratifiedKFold
from typing import List, Tuple, Sequence, Type  # Union
from abc import abstractmethod  # ,ABC
import settings


# Num = Union[int, float]

# The time series that you would get are such that the difference between two rows is 15 minutes.
# This is a global number that we used to prepare the data, so you would need it for different purposes.


####################################################################################################

class _DataCleaner:
    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df
        self._clean()

    def get_cleaned(self) -> pd.DataFrame:
        return self._df

    def _clean(self) -> None:
        self._remove_outliers()
        self._remove_duplicates()

    @abstractmethod
    def _remove_outliers(self) -> None:
        pass

    @abstractmethod
    def _remove_duplicates(self) -> None:
        pass


# @classmethod
#     def _remove_ids_with_missing_data(self, ids_with_data: pd.Index) -> None:
#         ids_with_data_mask = self._df.index.isin(ids_with_data, level=0)
#         self._df = self._df[ids_with_data_mask]

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
    # FEATURES_WITH_OUTLIERS = ['weight']

    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__(df)

    def _remove_outliers(self) -> None:
        keep_mask = np.abs(stats.zscore(self._df[settings.DataStructureMeals.FEATURES_WITH_OUTLIERS])) < 3
        self._df = self._df[keep_mask]

    def _remove_duplicates(self) -> None:
        indices = [settings.DataStructure.ID_HEADER, settings.DataStructure.DATE_HEADER]
        self._df = self._df.groupby(indices).sum()


####################################################################################################


class _DataEnricher:
    # GLUCOSE_VALUE_HEADER = 'GlucoseValue'

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df
        self._enrich()

    @classmethod
    def get_enriched(self) -> pd.DataFrame:
        return self._df

    @abstractmethod
    def _enrich(self) -> None:
        pass


class DataEnricherX(_DataEnricher):
    FEATURES_TO_ADD_ROLLING_WINDOW_INFORMATION = ['weight', 'carbohydrate_g', 'energy_kcal', 'totallipid_g']

    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__(df)

    def _enrich(self) -> None:
        self._enrich_with_rolling_window_mean(self.FEATURES_TO_ADD_ROLLING_WINDOW_INFORMATION)
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
        self._df = self._df[settings.DataStructureGlucose.GLUCOSE_VALUE_HEADER].join(new_df, how='outer')

    def _enrich_with_cyclic_time(self) -> None:
        """
        Encode the time in the day in a cyclic way (23:55 and 00:05 are 10 minutes apart, rather than 23 hours and 50 minutes apart)
        https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/
        """
        MINUTES_IN_A_DAY = 60 * 24
        time = self._df.index.get_level_values('Date').to_series().apply(lambda d: d.time())
        f = 2 * np.pi * time.apply(lambda t: t.hour * 60 + t.minute)
        self._df['sin_time'] = (np.sin(f) / MINUTES_IN_A_DAY).values
        self._df['cos_time'] = (np.cos(f) / MINUTES_IN_A_DAY).values

    @staticmethod
    def _enrich_with_future_timepoints(df: pd.DataFrame,
                                       n_future_time_points: int = 8,
                                       timepoints_resolution_in_minutes: int = 15) -> pd.DataFrame:
        """
        Extracting the m next time points (difference from time zero)
        :param n_future_time_points: number of future time points
        :return:
        """
        for i, g in enumerate(range(timepoints_resolution_in_minutes,
                                    timepoints_resolution_in_minutes * (n_future_time_points + 1),
                                    timepoints_resolution_in_minutes),
                              1):
            new_header = f'{settings.DataStructureGlucose.GLUCOSE_VALUE_HEADER} difference +%0.1dmin' % g
            df[new_header] = \
                df[settings.DataStructureGlucose.GLUCOSE_VALUE_HEADER].shift(-i) \
                - df[settings.DataStructureGlucose.GLUCOSE_VALUE_HEADER]
        return df.dropna(how='any', axis=0).drop(settings.DataStructureGlucose.GLUCOSE_VALUE_HEADER, axis=1)


####################################################################################################

class _DataProcessor:
    # ID_HEADER = 'id'
    # DATE_HEADER = 'Date'

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def get_processed(self) -> pd.DataFrame:
        return self._df.copy()

    def _clean(self, data_cleaner: Type[_DataCleaner]) -> None:
        data_cleaner = data_cleaner(self._df)
        self._df = data_cleaner.get_cleaned()


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

    @staticmethod
    def create_shifts(df, feature_name, n_previous_time_points=48):
        """
        Creating a data frame with columns corresponding to previous time points
        :param df: A pandas data frame
        :param n_previous_time_points: number of previous time points to shift
        :return:
        """
        for i, g in enumerate(
                range(
                    settings.DataStructureGlucose.SAMPLING_INTERVAL_IN_MINUTES,
                            settings.DataStructureGlucose.SAMPLING_INTERVAL_IN_MINUTES * (n_previous_time_points + 1),
                    settings.DataStructureGlucose.SAMPLING_INTERVAL_IN_MINUTES),
                1):
            df[f'{feature_name} -%0.1dmin' % g] = df[f'{feature_name}'].shift(i)
        return df.dropna(how='any', axis=0)

    def _filter_X_by_glucose_indices(self, glucose_value_header='GlucoseValue'):
        print('[Dataset] _filter_X_by_glucose_indices')
        self._all_X = self._all_X.dropna(subset=[glucose_value_header]).sort_index()

    @staticmethod
    def _normalize(df):
        print('[Dataset] _normalize')
        #         return (df - df.mean()) / (df.max() - df.min())
        return (df - df.mean()) / df.std()


####################################################################################################

class Dataset:
    # INDEX_COLUMNS = [0, 1]
    # HEADERS_WITH_DATES = ['Date']

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
        self._raw = pd.concat([glucose_df, meals_df], axis=1, join='inner').sort_index()

    def get_X_and_y(self,
                    n_previous_time_points: int = 48,
                    n_future_time_points: int = 8) \
            -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Returns X and y as dataframes. Contains only timepoints with enough past and future samples.
        """
        X = self._processed[settings.DataStructureGlucose.GLUCOSE_VALUE_HEADER].reset_index() \
            .groupby(settings.DataStructure.ID_HEADER) \
            .apply(DataProcessorX.create_shifts,
                   feature_name=settings.DataStructureGlucose.GLUCOSE_VALUE_HEADER,
                   n_previous_time_points=n_previous_time_points) \
            .set_index([settings.DataStructure.ID_HEADER, settings.DataStructure.DATE_HEADER])
        y = self._processed[settings.DataStructureGlucose.GLUCOSE_VALUE_HEADER].reset_index() \
            .groupby(settings.DataStructure.ID_HEADER) \
            .apply(DataEnricherX._enrich_with_future_timepoints,
                   n_future_time_points=8) \
            .set_index([settings.DataStructure.ID_HEADER, settings.DataStructure.DATE_HEADER])
        idx_intersection = X.index.intersection(y.index)
        return X.loc[idx_intersection], y.loc[idx_intersection]

    def get_multivariate_X_and_y(self,
                                 n_previous_time_points: int = 48,
                                 n_future_time_points: int = 8) \
            -> Sequence[Tuple[np.ndarray, np.ndarray]]:
        """
        Returns a 2-tuple of 3D np.ndarray with the following shapes:
        X's shape is (# of instances in the dataset, # past timepoints + 1, # of features),
        y's shape is (# of future timepoints, 1).
        """
        X, y = self.get_X_and_y(n_previous_time_points=n_previous_time_points,
                                n_future_time_points=n_future_time_points)
        multivariate_y = y.values.reshape(y.shape + (1,))

        shifted_Xs = []
        for feature_name in X.columns:
            shifted_X = X[feature_name].reset_index().groupby('id') \
                .apply(DataProcessorX.create_shifts,
                       feature_name=feature_name,
                       n_previous_time_points=48) \
                .set_index(['id', 'Date'])
            shifted_Xs.append(shifted_X)
        multivariate_X = np.dstack(shifted_Xs)
        return multivariate_X, multivariate_y


####################################################################################################

# class OLDDataset(object):
#     def __init__(self, raw_glucose, raw_meals):
#         self._raw_glucose = raw_glucose
#         self._raw_meals = raw_meals

#         self._clean_glucose = None
#         self._clean_meals = None

#         self._all_X = None

#         self._create_all_X()

#     @staticmethod
#     def load_raw_data(filename, dir_path=''):
#         """
#         Load a pandas data frame in the relevant format.
#         :param path: path to csv.
#         :return: the loaded data frame.
#         """
#         print('[Dataset] load_raw_data')
#         path = os.path.join(dir_path, filename)
#         print(f'loading the file: {path}')
#         return pd.read_csv(path, index_col=[0, 1], parse_dates=['Date'])

#     def _create_all_X(self, n_previous_time_points=48, n_future_time_points=8):
#         """
#         Given glucose and meals data, build the features needed for prediction.
#         :param X_glucose: A pandas data frame holding the glucose values.
#         :param X_meals: A pandas data frame holding the meals data.
#         :param n_previous_time_points:
#         :param n_future_time_points:
#         :return: The features needed for your prediction, and optionally also the relevant y arrays for training.
#         """
#         print('[Dataset] _create_all_X')
#         self._clean_raw_data()
#         self._all_X = self._clean_glucose.copy().join(self._clean_meals, how='outer').sort_index()
#         self._enrich_X()
#         self._filter_X_by_glucose_indices()

#     def _clean_raw_data(self):
#         print('[Dataset] _clean_raw_data')
#         cgm_cleaner = CgmCleaner(self._raw_glucose)
#         cgm_cleaner.remove_ids_with_no_meal_data(self._raw_meals)
#         cgm_cleaner.remove_outliers()
#         cgm_cleaner.zero_center()

#         meals_cleaner = MealsCleaner(self._raw_meals)
#         meals_cleaner.remove_ids_with_no_cgm_data(self._raw_glucose)
#         meals_cleaner.remove_outliers()
#         meals_cleaner.aggregate_by_timestamp()
# #         meals_cleaner.synchronize_timesteps_to_cgm_df(self._raw_glucose)

#         self._clean_glucose = cgm_cleaner.df
#         self._clean_meals = meals_cleaner.df

#     def _enrich_X(self):
#         print('[Dataset] _enrich_X')
#         self._enrich_X_with_rolling_mean()
#         self._enrich_X_with_cyclic_time()

#     def _enrich_X_with_rolling_mean(self,
#                                     headers=['weight', 'carbohydrate_g', 'energy_kcal', 'totallipid_g'],
#                                     win_size=8, win_type='exponential'):
#         print('[Dataset] _enrich_X_with_rolling_mean')
#         df_filled = self._all_X.fillna(0)
#         # windowing
#         new_headers = [f'{h}_{win_type}_{win_size}' for h in headers]
#         if win_type == 'triang':
#             win = df_filled[headers].rolling(win_size, win_type=win_type)
#         elif win_type == 'exponential':
#             win = df_filled[headers].ewm(span=win_size)
#         else:
#             raise NotImplementedError
#         new_df = win.mean()
#         # enrich original glucose df
#         new_df.columns = new_headers
#         self._all_X = self._clean_glucose.join(new_df, how='outer')

#     def _enrich_X_with_cyclic_time(self):
#         """
#        Encode the time in the day in a cyclic way (23:55 and 00:05 are 10 minutes apart, rather than 23 hours and 50 minutes apart)
#        https://ianlondon.github.io/blog/encoding-cyclical-features-24hour-time/
#        """
#         print('[Dataset] _enrich_X_with_cyclic_time')
#         MINUTES_IN_A_DAY = 60 * 24
#         time = self._all_X.index.get_level_values('Date').to_series().apply(lambda d: d.time())
#         self._all_X['sin_time'] = np.sin(
#             2 * np.pi * time.apply(lambda t: t.hour * 60 + t.minute) / MINUTES_IN_A_DAY).values
#         self._all_X['cos_time'] = np.cos(
#             2 * np.pi * time.apply(lambda t: t.hour * 60 + t.minute) / MINUTES_IN_A_DAY).values

#     def _filter_X_by_glucose_indices(self, glucose_value_header='GlucoseValue'):
#         print('[Dataset] _filter_X_by_glucose_indices')
#         self._all_X = self._all_X.dropna(subset=[glucose_value_header]).sort_index()

#     @staticmethod
#     def _normalize(df):
#         print('[Dataset] _normalize')
#         #         return (df - df.mean()) / (df.max() - df.min())
#         return (df - df.mean()) / df.std()

#     @staticmethod
#     def X_to_multivariate_X(X):
#         y_indices = Dataset.extract_y(X).index
#         shifted_Xs = []
#         for feature_name in X.columns:
#             shifted_X = X[feature_name].reset_index().groupby('id')\
#                                        .apply(Dataset._create_shifts,
#                                               feature_name=feature_name,
#                                               n_previous_time_points=48)\
#                                        .set_index(['id', 'Date'])
#             shifted_Xs.append(shifted_X.loc[y_indices])
#         return np.dstack(shifted_Xs)


#     @staticmethod
#     def X_to_multivariate_y(X):
#         y = Dataset.extract_y(X)
#         shape = y.shape
#         return y.values.reshape(shape + (1,))

def _get_splitted_X(self, num_of_train_to_split):
    print('[Dataset] _get_splitted_X')
    return self._all_X[:num_of_train_to_split], self._all_X[num_of_train_to_split:]


#     @staticmethod
#     def _enrich_with_future_timepoints(df, n_future_time_points=8):
#         """
#         Extracting the m next time points (difference from time zero)
#         :param n_future_time_points: number of future time points
#         :return:
#         """
#         for i, g in enumerate(
#                 range(DATA_RESOLUTION_MIN, DATA_RESOLUTION_MIN * (n_future_time_points + 1), DATA_RESOLUTION_MIN),
#                 1):
#             df['Glucose difference +%0.1dmin' % g] = df['GlucoseValue'].shift(-i) - df['GlucoseValue']
#         return df.dropna(how='any', axis=0).drop('GlucoseValue', axis=1)

#     @staticmethod
#     def extract_y(X, glucose_value_header='GlucoseValue'):
#         #         print('[Dataset] extract_y')
#         X_df = X[glucose_value_header].reset_index()\
#                                        .groupby('id')\
#                                        .apply(Dataset._create_shifts,
#                                               feature_name=glucose_value_header,
#                                               n_previous_time_points=48)\
#                                        .set_index(['id', 'Date'])
#         y_df = X[glucose_value_header].reset_index()\
#                                       .groupby('id')\
#                                       .apply(Dataset._enrich_with_future_timepoints,
#                                              n_future_time_points=8)\
#                                       .set_index(['id', 'Date'])
#         index_intersection = X_df.index.intersection(y_df.index)
# #         X_df = X_df.loc[index_intersection]
#         y_df = y_df.loc[index_intersection]
#         return y_df

#     @staticmethod
#     def _create_shifts(df, feature_name, n_previous_time_points=48):
#         """
#         Creating a data frame with columns corresponding to previous time points
#         :param df: A pandas data frame
#         :param n_previous_time_points: number of previous time points to shift
#         :return:
#         """
#         for i, g in enumerate(
#                 range(DATA_RESOLUTION_MIN, DATA_RESOLUTION_MIN * (n_previous_time_points + 1), DATA_RESOLUTION_MIN),
#                 1):
#             df[f'{feature_name} -%0.1dmin' % g] = df[f'{feature_name}'].shift(i)
#         return df.dropna(how='any', axis=0)

@staticmethod
def _get_shuffled_tf_dataset(multivariate_X, multivariate_y):
    print('[Dataset] _get_shuffled_tf_dataset')
    d = tf.data.Dataset.from_tensor_slices((multivariate_X, multivariate_y))
    return d.cache() \
        .shuffle(settings.TrainConfiguration.BUFFER_SIZE) \
        .batch(settings.TrainConfiguration.BATCH_SIZE) \
        .repeat()


#     def get_X(self):
#         print('[Dataset] get_X')
#         return self._all_X

def get_splited_and_shuffled_train_and_valid_datasets(self, num_of_train_to_split):
    print('[Dataset] get_splited_and_shuffled_train_and_valid_datasets')
    train_X, valid_X = self._get_splitted_X(num_of_train_to_split)
    # shuffled multivariate
    print('preparing training set...')
    multivariate_train_X = self.X_to_multivariate_X(train_X)
    multivariate_train_y = self.X_to_multivariate_y(train_X)

    print('preparing validation set...')
    multivariate_valid_X = self.X_to_multivariate_X(valid_X)
    multivariate_valid_y = self.X_to_multivariate_y(valid_X)

    train_dataset = self._get_shuffled_tf_dataset(multivariate_train_X, multivariate_train_y)
    valid_dataset = self._get_shuffled_tf_dataset(multivariate_valid_X, multivariate_valid_y)
    return train_dataset, valid_dataset


def get_num_of_features(self):
    print('[Dataset] get_num_of_features')
    return self._all_X.shape[1]


    # ################################################################################################################################################
    #
    # class Predictor(object):
    #     """
    #     This is where you should implement your predictor.
    #     The testing script calls the 'predict' function with the glucose and meals test data which you will need in order to
    #     build your features for prediction.
    #     You should implement this function as you wish, just do not change the function's signature (name, parameters).
    #     The other functions are here just as an example for you to have something to start with, you may implement whatever
    #     you wish however you see fit.
    #     """
    #
    #     #     def __init__(self, path2data):
    #     def __init__(self):
    #         """
    #         This constructor only gets the path to a folder where the training data frames are.
    #         :param path2data: a folder with your training data.
    #         """
    #         print('[Predictor] __init__')
    #         self._raw_glucose = None
    #         self._raw_meals = None
    #         self._dataset = None
    #         self.nn = None
    #
    #     def _eval(self, X):
    #         print('[Predictor] _eval')
    #         #         multivariate_x, multivariate_y = Dataset.X_to_multivariate_X_and_y(X)
    #         multivariate_x = Dataset.X_to_multivariate_X(X)
    #         multivariate_y = Dataset.X_to_multivariate_y(X)
    #         y_pred = self.nn(multivariate_x)
    #         y_pred = np.array(tf.cast(y_pred, 'float64'))
    #         indices = Dataset.extract_y(X).index
    #         return pd.DataFrame(y_pred, index=indices)
    #
    #     def predict(self, X_glucose, X_meals):
    #         """
    #         You must not change the signature of this function!!!
    #         You are given two data frames: glucose values and meals.
    #         For every timestamp (t) in X_glucose for which you have at least 12 hours (48 points) of past glucose and two
    #         hours (8 points) of future glucose, predict the difference in glucose values for the next 8 time stamps
    #         (t+15, t+30, ..., t+120).
    #
    #         :param X_glucose: A pandas data frame holding the glucose values in the format you trained on.
    #         :param X_meals: A pandas data frame holding the meals data in the format you trained on.
    #         :return: A numpy ndarray, sized (M x 8) holding your predictions for every valid row in X_glucose.
    #                  M is the number of valid rows in X_glucose (number of time stamps for which you have at least 12 hours
    #                  of past glucose values and 2 hours of future glucose values.
    #                  Every row in your final ndarray should correspond to:
    #                  (glucose[t+15min]-glucose[t], glucose[t+30min]-glucose[t], ..., glucose[t+120min]-glucose[t])
    #         """
    #         print('[Predictor] predict')
    #         self._raw_glucose = X_glucose
    #         self._raw_meals = X_meals
    #
    #         self.init_dataset(self._raw_glucose, self._raw_meals)
    #
    #         # self.load_nn_model('training_2020-03-15_13_00_45')
    #
    #         X = self._dataset.get_X()
    #         y_pred = self._eval(X)
    #         return y_pred
    #
    #     def _define_nn(self):
    #         """
    #         Define your neural network.
    #         :return: None
    #         """
    #         print('[Predictor] _define_nn')
    #         num_of_features = self._dataset.get_num_of_features()
    #         multi_step_model = tf.keras.models.Sequential()
    #         multi_step_model.add(tf.keras.layers.LSTM(32,
    #                                                   return_sequences=True,
    #                                                   input_shape=(49, num_of_features)))
    #         multi_step_model.add(tf.keras.layers.LSTM(16, activation='relu'))
    #         multi_step_model.add(tf.keras.layers.Dense(8))
    #
    #         multi_step_model.compile(optimizer=tf.keras.optimizers.Adam(0.001), loss='mse')
    #
    #         self.nn = multi_step_model
    #         return
    #
    #     def train_nn(self, num_of_train_to_split, num_of_epochs):
    #         print('[Predictor] train_nn')
    #         saving_path = 'training_{date:%Y-%m-%d_%H_%M_%S}'.format(date=pd.datetime.now())
    #
    #         train_dataset, valid_dataset = self._dataset.get_splited_and_shuffled_train_and_valid_datasets(
    #             num_of_train_to_split)
    #
    #         # checkpoints
    #         checkpoint_path = os.path.join(saving_path, 'checkpoints', 'cp-{epoch:04d}.ckpt')
    #         checkpoint_dir = os.path.dirname(checkpoint_path)
    #
    #         cp_callback = tf.keras.callbacks.ModelCheckpoint(
    #             filepath=checkpoint_path,
    #             verbose=1,
    #             save_weights_only=True)
    #
    #         fit_history = self.nn.fit(train_dataset,
    #                                   epochs=num_of_epochs,
    #                                   steps_per_epoch=EVALUATION_INTERVAL,
    #                                   validation_data=valid_dataset,
    #                                   validation_steps=50, callbacks=[cp_callback])
    #
    #         self.save_nn_model(os.path.join(saving_path, 'model'))
    #
    #         plot_train_history(fit_history, 'Multi-Step Training and validation loss')
    #         plt.show()
    #         plt.savefig(os.path.join(saving_path, 'loss.png'))
    #         #         return multi_step_history, val_data_multi, X_valid, y_valid
    #         return fit_history
    #
    #     def init_dataset(self, raw_glucose, raw_meals):
    #         print('[Predictor] init_dataset')
    #         self._dataset = Dataset(raw_glucose, raw_meals)
    #
    #     def save_nn_model(self, saved_model_path):
    #         print('[Predictor] save_nn_model')
    #         self.nn.save(saved_model_path)
    #         pass
    #
    #     def load_nn_model(self, model_path, checkpoint_num=None):
    #         """
    #         Load your trained neural network.
    #         :return:
    #         """
    #         print('[Predictor] load_nn_model')
    #         self.nn = tf.keras.models.load_model(os.path.join(model_path, 'model'))
    #         if checkpoint_num:
    #             self.nn.load_weights(os.path.join(model_path, 'checkpoints',
    #                                               f'cp-{checkpoint_num:04d}.ckpt'))
    #
    #     def train_new_nn(self, num_of_train_to_split, num_of_epochs):
    #         print('[Predictor] train_new_nn')
    #         self._define_nn()
    #         fit_history = self.train_nn(num_of_train_to_split=num_of_train_to_split,
    #                                     num_of_epochs=num_of_epochs)
    #         return fit_history
    #
    #
    # ####################################################################################################

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
