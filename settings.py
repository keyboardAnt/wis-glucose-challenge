import tensorflow as tf


class Files:
    DATA_DIR_PATH = 'data'

    RAW_DATA_DIR_NAME = 'raw'
    RAW_GLUCOSE_FILENAME = 'GlucoseValues.csv'
    RAW_MEALS_FILENAME = 'Meals.csv'

    PROCESSED_DATA_DIR_NAME = 'processed'
    PROCESSED_DATASET_X_FILENAME = 'ProcessedX.csv'
    PROCESSED_DATASET_GLUCOSE_FILENAME = 'ProcessedGlucoseValues.csv'
    PROCESSED_DATASET_MEALS_FILENAME = 'ProcessedMeals.csv'

    CHECKPOINTS_DIR_NAME = 'checkpoints'


class DataStructure:
    INDEX_COLUMNS = [0, 1]
    ID_HEADER = 'id'
    DATE_HEADER = 'Date'
    HEADERS_WITH_DATES = [DATE_HEADER]


class DataStructureGlucose(DataStructure):
    SAMPLING_INTERVAL_IN_MINUTES = 15
    GLUCOSE_VALUE_HEADER = 'GlucoseValue'
    GLUCOSE_CORRELATED_FEATURES = ['weight', 'carbohydrate_g', 'energy_kcal', 'totallipid_g']


class DataStructureMeals(DataStructure):
    FEATURES_WITH_OUTLIERS = ['weight']


# class DataStructureX(DataStructureGlucose, DataStructureMeals):
#     pass


class NN:
    MODEL = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(32, return_sequences=True, input_shape=(49, 7)),
        tf.keras.layers.LSTM(16, activation='relu'),
        tf.keras.layers.Dense(8)
    ])
    OPTIMIZER = 'adam'
    LOSS = 'mse'


class TrainingConfiguration:
    BATCH_SIZE = 256
    TRAIN_SPLIT = 690000
    STEP = 1
    BUFFER_SIZE = 1000
    EVALUATION_INTERVAL = 200
    VALIDATION_STEPS = 50
    NUM_OF_EPOCHS = 1
    CROSS_VALIDATION_NUM_OF_FOLDS = 3
