import tensorflow as tf
from typing import Type


class Files:
    DATA_DIR_PATH = 'data'

    RAW_DATA_DIR_NAME = 'raw'
    RAW_GLUCOSE_FILENAME = 'test_GlucoseValues.csv' #'GlucoseValues.csv'
    RAW_MEALS_FILENAME = 'test_Meals.csv' #'Meals.csv'
    FOOD_NAMES_FILENAME = 'food_names.csv'

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
    ID_HEADER = 'RegistrationCode' #'id'
    DATE_HEADER = 'Date'
    FOOD_ID_HEADER = 'food_id'
    HEADERS_WITH_DATES = [DATE_HEADER]


class DataStructureGlucose(DataStructure):
    SAMPLING_INTERVAL_IN_MINUTES = 15
    GLUCOSE_VALUE_HEADER = 'GlucoseValue'
    GLUCOSE_CORRELATED_FEATURES = ['weight',
                                   'unit_id',
                                   'alcohol_g',
                                   'caffeine_mg',
                                   'calcium_mg',
                                   'carbohydrate_g',
                                   'cholesterol_mg',
                                   'energy_kcal',
                                   'magnesium_mg',
                                   'niacin_mg',
                                   'protein_g',
                                   'sodium_mg',
                                   'sugarstotal_g',
                                   'thiamin_mg',
                                   'totaldietaryfiber_g',
                                   'totallipid_g',
                                   'totalmonounsaturatedfattyacids_g',
                                   'totalpolyunsaturatedfattyacids_g',
                                   'totalsaturatedfattyacids_g',
                                   'totaltransfattyacids_g',
                                   'vitaminc_mg',
                                   'vitamind_iu',
                                   'vitamine_mg',
                                   'water_g',
                                   'zinc_mg']
    # GLUCOSE_CORRELATED_FEATURES = ['weight', 'carbohydrate_g', 'energy_kcal', 'totallipid_g']


class DataStructureMeals(DataStructure):
    FEATURES_WITH_OUTLIERS = ['weight']


class Challenge:
    NUM_OF_PAST_TIMEPOINTS = 12 * 60 // DataStructureGlucose.SAMPLING_INTERVAL_IN_MINUTES
    NUM_OF_FUTURE_TIMEPOINTS = 2 * 60 // DataStructureGlucose.SAMPLING_INTERVAL_IN_MINUTES


class NN:
    @staticmethod
    def get_model() -> Type[tf.keras.models.Model]:
        return tf.keras.models.Sequential([
            tf.keras.layers.LSTM(32, return_sequences=True, input_shape=(1 + Challenge.NUM_OF_PAST_TIMEPOINTS, 28)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.LSTM(16, dropout=.5),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(8)
    ])
    # return
    # MODEL = tf.keras.models.Sequential([
    #     tf.keras.layers.LSTM(32, return_sequences=True, input_shape=(1 + Challenge.NUM_OF_PAST_TIMEPOINTS, 7)),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.LSTM(16, dropout=.1),
    #     tf.keras.layers.BatchNormalization(),
    #     tf.keras.layers.Dense(8)
    # ])
    OPTIMIZER = 'sgd'
    LOSS = 'mse'

    class BEST:
        LOGS_DIR_NAME = 'best'
        CHECKPOINT_NUM = 37


class TrainingConfiguration:
    BUFFER_SIZE = 100000
    BATCH_SIZE = 512
    # BATCH_SIZE = 64
    NUM_OF_EPOCHS = 100
    CROSS_VALIDATION_NUM_OF_FOLDS = 10
    STEPS_PER_EPOCH = 50
    VALIDATION_STEPS = 100
