class DataStructure:
    INDEX_COLUMNS = [0, 1]
    ID_HEADER = 'id'
    DATE_HEADER = 'Date'
    HEADERS_WITH_DATES = [DATE_HEADER]


class DataStructureGlucose(DataStructure):
    SAMPLING_INTERVAL_IN_MINUTES = 15
    GLUCOSE_VALUE_HEADER = 'GlucoseValue'


class DataStructureMeals(DataStructure):
    FEATURES_WITH_OUTLIERS = ['weight']


class DataStructureX(DataStructureGlucose, DataStructureMeals):
    pass


class TrainConfiguration:
    BATCH_SIZE = 256
    TRAIN_SPLIT = 690000
    STEP = 1
    BUFFER_SIZE = 1000
    EVALUATION_INTERVAL = 200
    NUM_OF_EPOCHS = 500


class Files:
    RAW_DATA_DIR_PATH = 'raw_data'
    RAW_GLUCOSE_FILENAME = 'GlucoseValues.csv'
    RAW_MEALS_FILENAME = 'Meals.csv'

    PROCESSED_DATA_DIR_PATH = 'processed_data'
    PROCESSED_DATASET_X_FILENAME = 'ProcessedDatasetX.csv'
