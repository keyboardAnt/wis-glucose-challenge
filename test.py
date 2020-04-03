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
    # Paths
    # raw_data_dir_path = os.path.join(
    #     settings.Files.DATA_DIR_PATH,
    #     settings.Files.RAW_DATA_DIR_NAME
    # )
    processed_data_dir_path = os.path.join(
        settings.Files.DATA_DIR_PATH,
        settings.Files.PROCESSED_DATA_DIR_NAME
    )
    # # Load & process datasets
    # # Glucose
    # glucose_dataset = Dataset()
    # glucose_dataset.load_raw(
    #     settings.Files.RAW_GLUCOSE_FILENAME,
    #     raw_data_dir_path
    # )
    # glucose_dataset.process(DataProcessorGlucose)
    # # Meals
    # meals_dataset = Dataset()
    # meals_dataset.load_raw(
    #     settings.Files.RAW_MEALS_FILENAME,
    #     raw_data_dir_path
    # )
    # meals_dataset.process(DataProcessorMeals)
    #
    # # NOTE: It's possible to predict by calling to `predictor.predict`
    # # y_pred = predictor.predict(X_glucose=glucose_dataset.get_raw(),
    # #                            X_meals=meals_dataset.get_raw())

    # # Build & process datasetX
    dataset = DatasetX()
    # dataset.build_raw_X_from_glucose_and_meals_datasets(
    #     glucose_dataset=glucose_dataset,
    #     meals_dataset=meals_dataset
    # )
    # dataset.process(DataProcessorX)
    # # Store processed
    # dataset.save_processed(
    #     settings.Files.PROCESSED_DATASET_X_FILENAME,
    #     processed_data_dir_path
    # )
    dataset.load_processed(
        settings.Files.PROCESSED_DATASET_X_FILENAME,
        processed_data_dir_path
    )
    print(
        'dataset.get_processed_shape()',
        dataset.get_processed_shape()
    )
    # Load best predictor model
    predictor = Predictor()
    predictor.load(
        settings.NN.BEST.LOGS_DIR_NAME,
        settings.NN.BEST.CHECKPOINT_NUM
    )
    # Predict & store
    multivariate_X, y_true = dataset.get_multivariate_X_and_y_true()
    # y_true = pd.DataFrame(y_true)
    print('y_true')
    print(y_true)
    y_true.to_csv(
        'y_true.csv',
        header=False,
        index=False
    )
    # y_true = pd.read_csv('y_true.csv', header=None)
    y_pred = predictor._predict(multivariate_X, multi_index=y_true.index)
    print('y_pred')
    print(y_pred)
    y_pred.to_csv(
        'y_pred.csv',
        header=False,
        index=False
    )
    # y_pred = pd.read_csv('y_pred.csv', header=None)
    # Evaluate
    mean_pearson = compute_mean_pearson(
        y_true=y_true,
        y_pred=y_pred
    )
    print(
        'mean_pearson',
        mean_pearson
    )
