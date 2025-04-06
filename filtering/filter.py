#%%

import pandas as pd
import numpy as np
import os
import pickle


def get_df_kg(df_kgg, df_type):
    df_kg_right = df_kgg.copy()

    df_kg_right['entity'] = df_kg_right['entity_1']
    df_kg_right['knowledge'] = ['[MASK] ' for _ in range(len(df_kg_right))] + df_kg_right['relations'] + [' ' for _ in range(len(df_kg_right))] + df_kg_right['entity_2']

    df_kg_right = df_kg_right.iloc[:, [3, 4]]

    df_kg_left = df_kgg.copy()

    df_kg_left['entity'] = df_kg_left['entity_2']
    df_kg_left['knowledge'] = df_kg_left['entity_1'] + [' ' for _ in range(len(df_kg_left))] + df_kg_left['relations'] + [' [MASK]' for _ in range(len(df_kg_left))]
    df_kg_left = df_kg_left.iloc[:, [3, 4]]

    df_kg_assioc = pd.concat([df_kg_right, df_kg_left])

    df_kg_assioc = df_kg_assioc.merge(df_type, on='entity', how='left')

    df_kg = pd.DataFrame(df_kg_assioc['knowledge'].unique(), columns=['knowledge'])
    count_types = pd.DataFrame(df_kg_assioc.groupby('knowledge')['type'].nunique())
    count_types.reset_index(inplace=True)
    df_kg = df_kg.merge(count_types, on='knowledge', how='left')
    df_kg.rename(columns={'type': 'count_types'}, inplace=True)
    count_entities = pd.DataFrame(df_kg_assioc.groupby('knowledge')['entity'].nunique())
    count_entities.reset_index(inplace=True)
    df_kg = df_kg.merge(count_entities, on='knowledge', how='left')
    df_kg.rename(columns={'entity': 'count_entities'}, inplace=True)

    return df_kg, df_kg_assioc


def get_kl_divergence(kg_path, et_path):
    # Charger les fichiers de connaissances et d'entités
    df_kgg = pd.read_csv(kg_path, sep='\t', names=['entity_1', 'relations', 'entity_2'])

    df_type = pd.read_csv(et_path, sep='\t', names=['entity', 'type'])
    
    df_kg, df_kg_assioc = get_df_kg(df_kgg, df_type)

    # Calculer la distribution des types pour chaque connaissance
    type_counts = df_kg_assioc.groupby(['knowledge', 'type'])['entity'].nunique().reset_index()
    type_counts.rename(columns={'entity': 'count'}, inplace=True)

    # Fusion avec df_kg pour obtenir count_entities
    type_counts = type_counts.merge(df_kg[['knowledge', 'count_entities']], on='knowledge', how='left')

    # Calcul des probabilités p(t | knowledge)
    type_counts['p_knowledge'] = type_counts['count'] / type_counts['count_entities']

    # Calculer la distribution du type pour l'ensemble des types
    type_counts_all = df_type.groupby('type')['entity'].nunique().reset_index()
    type_counts_all.rename(columns={'entity': 'count'}, inplace=True)
    type_counts_all['p_all'] = type_counts_all['count'] / type_counts_all['count'].sum()

    

    # Fusionner les deux distributions pour obtenir p(t | knowledge) et p(t)
    type_counts = type_counts.merge(type_counts_all, on='type', how='left')

    # Éviter log(0) en filtrant les probabilités nulles
    type_counts = type_counts[(type_counts['p_knowledge'] > 0) & (type_counts['p_all'] > 0)]

    # Calculer la divergence KL pour chaque connaissance
    type_counts['kl_divergence'] = type_counts['p_knowledge'] * np.log(type_counts['p_knowledge'] / type_counts['p_all'])

    # Regrouper les valeurs de divergence KL par connaissance
    kg_kl_divergence = type_counts.groupby('knowledge')['kl_divergence'].sum().reset_index()

    # Fusionner avec df_kg pour garder toutes les connaissances même celles sans divergence KL
    df_kg = df_kg.merge(kg_kl_divergence, on='knowledge', how='left')

    # Remplacer les NaN (aucune divergence KL calculée) par 0
    df_kg['kl_divergence'] = df_kg['kl_divergence'].fillna(0)

    return df_kg

def knowledge_to_id(df_kg, KG_path):
    df_rel = pd.read_csv(os.path.join(KG_path, 'relations.tsv'), sep='\t', header=None, names=['relation', 'id'])
    df_ent = pd.read_csv(os.path.join(KG_path, 'entities.tsv'), sep='\t', header=None, names=['entity', 'id'])
    df_clust = pd.read_csv(os.path.join(KG_path, 'clusters.tsv'), sep='\t', header=None, names=['cluster', 'id'], keep_default_na=False)

    clust_size = len(df_clust)
    rel_size = len(df_rel)


    df_ent = pd.concat([df_ent, pd.DataFrame([{'entity': '[MASK]', 'id': -1}])], ignore_index=True)


    df_kg_filtered = pd.DataFrame(df_kg['knowledge'].apply(lambda x : x.split(' ')).to_list(), columns=['entity_1', 'relation', 'entity_2'])

    df_kg_id = (
        df_kg_filtered.rename(columns={"entity_1": "entity", "entity_2": "entity_2"})
            .merge(df_ent, on="entity")
            .rename(columns={"id": "id_1", "entity": "entity_1", "entity_2": "entity"})
            .merge(df_ent, on="entity")
            .rename(columns={"id": "id_2", "entity": "entity_2"})
            .merge(df_rel, on="relation")
            .rename(columns={"id": "rel_id"})
            [["id_1", "rel_id", "id_2"]]
    )

    def invert_id(row):
        if row['id_2'] == -1:
            row['id_1'], row['id_2'] = row['id_2'], row['id_1']
            row['rel_id'] += clust_size + rel_size
        
        return row
    df_kg_id = df_kg_id.apply(invert_id, axis=1)

    return df_kg_id.values.tolist()

def filter_pkl(pkl_paths, pkl_output_paths, kg_filter):
    """
    Filter pkl files based on the provided paths.

    Args:
        pkl_paths (list): List of paths to the pkl files to be filtered.
        pkl_output_paths (list): List of paths where the filtered pkl files will be saved.
    """
    for pkl_path, pkl_output_path in zip(pkl_paths, pkl_output_paths):
        print(f'Filtering {pkl_path}...')
        num_filtered = 0
        new_data = []
        with open(pkl_path, 'rb') as f:
            data = pickle.load(f)
            for et, kg, ent in data:
                new_kg = []
                for kg_triple in kg:
                    if [-1, kg_triple[1], kg_triple[2]] not in kg_filter:
                        new_kg.append(kg_triple)
                    else:
                        num_filtered += 1
                if len(new_kg) == 0:
                    continue
                new_data.append((et, new_kg, ent))

        print(f'Filtered {num_filtered} triples.')
        with open(pkl_output_path, 'wb') as f:
            pickle.dump(new_data, f)
                        



    


# %%


dir = '../data_sampled'

kg_path = f'{dir}/KG_train.txt'
et_path = f'{dir}/ET_train.txt'
df_kg = get_kl_divergence(kg_path, et_path)

df_kg_filter = df_kg[df_kg['kl_divergence'] > df_kg['kl_divergence'].quantile(0.95)]

kg_filter = knowledge_to_id(df_kg_filter, '../data/YAGO43kET/')


filter_pkl(
    pkl_paths=[f'{dir}/LMET_train.pkl', f'{dir}/LMET_valid.pkl', f'{dir}/LMET_test.pkl'],
    pkl_output_paths=[f'{dir}/LMET_train_filtered.pkl', f'{dir}/LMET_valid_filtered.pkl', f'{dir}/LMET_test_filtered.pkl'],
    kg_filter=kg_filter
)


# %%

with open(f'{dir}/LMET_train_filtered.pkl', 'rb') as f:
    data = pickle.load(f)


# %%
