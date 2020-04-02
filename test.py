from classes import *
import os
import pandas as pd
from scipy.stats import pearsonr


def compute_mean_pearson(
        y_true: pd.DataFrame,
        y_pred: pd.DataFrame,
        individual_index_name: str = settings.DataStructure.ID_HEADER,
        n_future_time_points: int = settings.Challenge.NUM_OF_FUTURE_TIMEPOINTS):
    """
    This function takes the true glucose values and the predicted ones, flattens the data per individual and then
    computed the Pearson correlation between the two vectors per individual.

    **This is how we will evaluate your predictions, you may use this function in your code**

    :param y_true: an M by n_future_time_points data frame holding the true glucose values
    :param y_pred: an M by n_future_time_points data frame holding the predicted glucose values
    :param individual_index_name: the name of the individual's indeces, default is 'id'
    :param n_future_time_points: number of future time points to predict, default is 8
    :return: the mean Pearson correlation
    """
    # making sure y_true and y_pred are of the same size
    assert y_true.shape == y_pred.shape
    # making sure y_true and y_pred share the same exact indeces and index names
    assert (y_true.index == y_pred.index).all() and y_true.index.names == y_pred.index.names
    # making sure that individual_index_name is a part of the index of both dataframes
    assert individual_index_name in y_true.index.names and individual_index_name in y_pred.index.names

    # concat data frames
    joined_df = pd.concat((y_true, y_pred), axis=1)
    return joined_df.groupby(individual_index_name) \
        .apply(lambda x: pearsonr(x.iloc[:, :n_future_time_points].values.ravel(),
                                  x.iloc[:, n_future_time_points:].values.ravel())[0]).mean()


if __name__ == "__main__":
    raw_data_dir_path = os.path.join(settings.Files.DATA_DIR_PATH, settings.Files.RAW_DATA_DIR_NAME)
    glucose_dataset = Dataset()
    glucose_dataset.load_raw(settings.Files.RAW_GLUCOSE_FILENAME, raw_data_dir_path)
    meals_dataset = Dataset()
    meals_dataset.load_raw(settings.Files.RAW_MEALS_FILENAME, raw_data_dir_path)
    predictor = Predictor()
    y_pred = predictor.predict(X_glucose=glucose_dataset.get_raw(),
                               X_meals=meals_dataset.get_raw())
    print(y_pred)
    # compute_mean_pearson(y_true=y_, y_pred=y_pred)
