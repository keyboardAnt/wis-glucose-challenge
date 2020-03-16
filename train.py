from classes import *
import os

if __name__ == "__main__":
    raw_data_dir_path = os.path.join(settings.Files.DATA_DIR_PATH, settings.Files.RAW_DATA_DIR_NAME)
    processed_data_dir_path = os.path.join(settings.Files.DATA_DIR_PATH, settings.Files.PROCESSED_DATA_DIR_NAME)

    # glucose_dataset = Dataset()
    # glucose_dataset.load_raw(settings.Files.RAW_GLUCOSE_FILENAME, raw_data_dir_path)
    # glucose_dataset.process(DataProcessorGlucose)
    #
    # meals_dataset = Dataset()
    # meals_dataset.load_raw(settings.Files.RAW_MEALS_FILENAME, raw_data_dir_path)
    # meals_dataset.process(DataProcessorMeals)

    dataset = DatasetX()
    # dataset.build_X_raw_from_glucose_and_meals_datasets(glucose_dataset=glucose_dataset, meals_dataset=meals_dataset)
    # dataset.process(DataProcessorX)
    # dataset.save_processed(settings.Files.PROCESSED_DATASET_X_FILENAME, processed_data_dir_path)
    dataset.load_processed(settings.Files.PROCESSED_DATASET_X_FILENAME, processed_data_dir_path)

    print(dataset.get_processed())

    # predictor = Predictor()
    # # predictor.load()
    # num_of_features = dataset.get_processed_shape()[1]
    # predictor.reset(num_of_features)
    # trainer = Trainer(predictor)
    # history = trainer.train(dataset, settings.TrainConfiguration.NUM_OF_EPOCHS)
    # trainer.plot_history(history)
    # # predictor.save()
    #
    # evaluator = Evaluator(predictor)
    # evaluator.calc_loss(loss_fn(y_true, y_pred))
    # evaluator.plot_correlation()
