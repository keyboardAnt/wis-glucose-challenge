from classes import *
import os

if __name__ == "__main__":
    # raw_data_dir_path = os.path.join(settings.Files.DATA_DIR_PATH, settings.Files.RAW_DATA_DIR_NAME)
    processed_data_dir_path = os.path.join(settings.Files.DATA_DIR_PATH, settings.Files.PROCESSED_DATA_DIR_NAME)

    # glucose_dataset = Dataset()
    # # glucose_dataset.load_raw(settings.Files.RAW_GLUCOSE_FILENAME, raw_data_dir_path)
    # # glucose_dataset.process(DataProcessorGlucose)
    # # glucose_dataset.save_processed(settings.Files.PROCESSED_DATASET_GLUCOSE_FILENAME,
    # #                                processed_data_dir_path)
    # glucose_dataset.load_processed(settings.Files.PROCESSED_DATASET_GLUCOSE_FILENAME,
    #                                processed_data_dir_path)
    # print('glucose_dataset.get_processed_shape()', glucose_dataset.get_processed_shape())
    #
    # meals_dataset = Dataset()
    # # meals_dataset.load_raw(settings.Files.RAW_MEALS_FILENAME, raw_data_dir_path)
    # # meals_dataset.process(DataProcessorMeals)
    # # meals_dataset.save_processed(settings.Files.PROCESSED_DATASET_MEALS_FILENAME,
    # #                              processed_data_dir_path)
    # meals_dataset.load_processed(settings.Files.PROCESSED_DATASET_MEALS_FILENAME,
    #                              processed_data_dir_path)
    # print('meals_dataset.get_processed_shape()', meals_dataset.get_processed_shape())

    dataset = DatasetX()
    # dataset.build_raw_X_from_glucose_and_meals_datasets(glucose_dataset=glucose_dataset, meals_dataset=meals_dataset)
    # dataset.process(DataProcessorX)
    # dataset.save_processed(settings.Files.PROCESSED_DATASET_X_FILENAME, processed_data_dir_path)
    dataset.load_processed(settings.Files.PROCESSED_DATASET_X_FILENAME, processed_data_dir_path)
    print('dataset.get_processed_shape()', dataset.get_processed_shape())

    predictor = Predictor()
    # predictor.reset()
    predictor.load(settings.NN.BEST.LOGS_DIR_NAME, settings.NN.BEST.CHECKPOINT_NUM)

    trainer = Trainer(predictor)
    history_of_all_folds = trainer.train(dataset, settings.TrainingConfiguration.NUM_OF_EPOCHS)
    trainer.plot_history(history_of_all_folds[-1])
    predictor.save()

    # evaluator = Evaluator(predictor)
    # evaluator.calc_loss(loss_fn(y_true, y_pred))
    # evaluator.plot_correlation()
