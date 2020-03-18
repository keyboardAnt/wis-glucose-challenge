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

    LOGS_DIR_NAME = 'logs'
    SAVED_MODEL_DIR_NAME = 'model'
    CHECKPOINTS_DIR_NAME = 'checkpoints'
    FIT_HISTORY_DIR_NAME = 'fit_history'


class DataStructure:
    INDEX_COLUMNS = [0, 1]
    ID_HEADER = 'id'
    DATE_HEADER = 'Date'
    HEADERS_WITH_DATES = [DATE_HEADER]


class DataStructureGlucose(DataStructure):
    SAMPLING_INTERVAL_IN_MINUTES = 15
    GLUCOSE_VALUE_HEADER = 'GlucoseValue'
    GLUCOSE_CORRELATED_FEATURES = ['weight', 'carbohydrate_g', 'energy_kcal', 'totallipid_g', 'caffeine_mg']


class DataStructureMeals(DataStructure):
    FEATURES_WITH_OUTLIERS = ['weight']


class Challenge:
    NUM_OF_PAST_TIMEPOINTS = 12 * 60 // DataStructureGlucose.SAMPLING_INTERVAL_IN_MINUTES
    NUM_OF_FUTURE_TIMEPOINTS = 2 * 60 // DataStructureGlucose.SAMPLING_INTERVAL_IN_MINUTES


class NN:
    MODEL = tf.keras.models.Sequential([
        tf.keras.layers.LSTM(32, return_sequences=True, input_shape=(49, 7)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LSTM(16, dropout=.1),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(8)
    ])
    OPTIMIZER = 'adam'
    LOSS = 'mse'

    class BEST:
        LOGS_DIR_NAME = 'best_model'
        CHECKPOINT_NUM = 17


class TrainingConfiguration:
    BATCH_SIZE = 256
    NUM_OF_EPOCHS = 50
    # STEP = 1
    # BUFFER_SIZE = 1000
    CROSS_VALIDATION_NUM_OF_FOLDS = 10
    # EVALUATION_INTERVAL = 200
    STEPS_PER_EPOCH = 200
    VALIDATION_STEPS = 50
