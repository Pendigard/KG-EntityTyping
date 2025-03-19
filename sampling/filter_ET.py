#%%
import pandas as pd


def filter_ET(ET_path, KG_sampled_path, ET_output_path):
    """
    @brief Filter ET file to keep only entities that are in the sampled KG
    @param ET_path : path to the ET file
    @param KG_sampled_path : path to the sampled KG
    """
    df_kg = pd.read_csv(KG_sampled_path, sep='\t', header=None, names=['entity', 'relation', 'entity_2'])
    entities = set(df_kg['entity'].values).union(set(df_kg['entity_2'].values))
    df_et = pd.read_csv(ET_path, sep='\t', header=None, names=['entity', 'type'])
    df_et = df_et[df_et['entity'].isin(entities)]
    df_et.to_csv(ET_output_path, sep='\t', header=False, index=False)

#%%

ET_types = ['test', 'train', 'valid']

for ET_type in ET_types:
    ET_path = f'../data/YAGO43kET/ET_{ET_type}.txt'
    KG_sampled_path = '../data_sampled/KG_train_sampled.txt'
    ET_output_path = f'../data_sampled/ET_{ET_type}_sampled.txt'

    filter_ET(ET_path, KG_sampled_path, ET_output_path)
# %%
