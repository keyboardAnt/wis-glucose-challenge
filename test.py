from classes import *
import os

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
