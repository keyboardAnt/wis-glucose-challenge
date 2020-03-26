import pandas as pd
import settings
from os.path import join
from sklearn.manifold import TSNE


not_numric_columns = ['meal_type', 'shortname_eng']
weight_columns = ['weight', 'unit_id']

raw_data_dir_path = join('..',
                         settings.Files.DATA_DIR_PATH,
                         settings.Files.RAW_DATA_DIR_NAME)
raw_meals_data_file_path = join(raw_data_dir_path,
                                settings.Files.RAW_MEALS_FILENAME)
df = pd.read_csv(raw_meals_data_file_path)
df = df.drop(columns=[settings.DataStructure.ID_HEADER,
                      settings.DataStructure.DATE_HEADER])\
       .drop_duplicates(settings.DataStructure.FOOD_ID_HEADER)\
       .set_index(settings.DataStructure.FOOD_ID_HEADER)
food_names_file_path = join(raw_data_dir_path,
                            settings.Files.FOOD_NAMES_FILENAME)
food_names = pd.read_csv(food_names_file_path, index_col=settings.DataStructure.FOOD_ID_HEADER)
df = df.join(food_names).dropna()
df = df.drop(columns=not_numric_columns).div(df['weight'], axis=0).join(df[not_numric_columns])
df[['shortname_eng']].to_csv('food_labels.tsv', sep='\t', index=False, header=False)

tsne_res = TSNE(n_components=3).fit_transform(df.drop(columns=not_numric_columns + weight_columns))
tsne_df = pd.DataFrame(tsne_res, columns=[f'tSNE{i}' for i in range(3)], index=df.index)
tsne_df.to_csv('food_data_tsne.tsv', sep='\t', index=False, header=False)