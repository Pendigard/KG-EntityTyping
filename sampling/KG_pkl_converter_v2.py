#%%
import pickle
import pandas as pd
import os
from tqdm import tqdm

tqdm.pandas() 


def get_cluster_name_from_type(type_name):
    if len(type_name.split('_')) < 2:
        return type_name
    cluster_name = type_name.split('_')[1]
    return cluster_name


def convert_to_pkl(KG_path, ET_file, output_file, KG_train='KG_train.txt', ET_train='ET_train.txt', get_clust_name_fonc=get_cluster_name_from_type):
    """
    @brief convert a knowledge graph into a pkl accepted by SSET
    @param KG_path : folder where clusters.tsv, entities.tsv and type.tsv are present
    @param ET_file : ET file to convert (Formatted as -> entity\ttype)
    @param get_clust_name_fonc : function to get cluster from type must take type and df_clust as input
    @param output_file : pkl generated
    """
    df_et = pd.read_csv(ET_file, sep='\t', header=None, names=['entity', 'type'])

    df_et_train = pd.read_csv(ET_train, sep='\t', header=None, names=['entity', 'type'])
    df_et = df_et[df_et['entity'].isin(set(df_et_train['entity'].values))]

    rel_path = os.path.join(KG_path, 'relations.tsv')
    df_rel = pd.read_csv(rel_path, sep='\t', header=None, names=['relation', 'id'])
    rel_count = len(df_rel)

    ent_path = os.path.join(KG_path, 'entities.tsv')
    df_ent = pd.read_csv(ent_path, sep='\t', header=None, names=['entity', 'id'])
    ent_count = len(df_ent)

    ent_series = df_ent.join(df_et.set_index('entity'), on='entity', how='inner')['entity'].unique()

    ent_series_id = df_ent.join(df_et.set_index('entity'), on='entity', how='inner')['id'].unique()

    type_path = os.path.join(KG_path, 'types.tsv')
    df_type = pd.read_csv(type_path, sep='\t', header=None, names=['type', 'id'])
    df_type['id'] += ent_count

    clust_path = os.path.join(KG_path, 'clusters.tsv')
    df_clust = pd.read_csv(clust_path, sep='\t', header=None, names=['cluster', 'id'], keep_default_na=False)
    df_clust['id'] += rel_count
    
    
    KG_train_path = os.path.join(KG_train)
    df_kg = pd.read_csv(KG_train_path, sep='\t', header=None, names=['entity', 'relation', 'entity_2'])
    print(df_kg.shape)
    
    df_ent = df_ent[df_ent['entity'].isin(set(df_kg['entity'].values).union(set(df_kg['entity_2'].values)))]

    df_kg_id = (
        df_kg.merge(df_ent, on="entity")
            .rename(columns={"id": "id_1", "entity": "entity_1", "entity_2": "entity"})
            .merge(df_ent, on="entity")
            .rename(columns={"id": "id_2", "entity": "entity_2"})
            .merge(df_rel, on="relation")
            .rename(columns={"id": "rel_id"})
            [["id_1", "rel_id", "id_2"]]
    )
    print(df_kg_id.shape)

    df_et = df_et[df_et['type'].isin(set(df_type['type'].values))]

    df_type = df_type[df_type['type'].isin(set(df_et['type'].values))]

    # df_ent = df_ent[df_ent['entity'].isin(set(df_et['entity'].values))]
    
    df_et_id = (
        df_et.merge(df_type.assign(clust_id=df_type['type'].progress_apply(lambda t: df_clust[df_clust['cluster'] == get_clust_name_fonc(t)]['id'].iloc[0])), on='type')
        .rename(columns={'id': 'type_id'})[['entity', 'clust_id', 'type_id']]
        .merge(df_ent, on='entity')
        .rename(columns={'id': 'ent_id'})[['ent_id', 'clust_id', 'type_id']]
    )

    triplets_list = [
    (df_et_id[(df_et_id['ent_id'] == ent_id)].values.tolist()
     , df_kg_id[(df_kg_id['id_1'] == ent_id) | (df_kg_id['id_2'] == ent_id)].values.tolist()
     , ent_id)    
    for ent_id in tqdm(ent_series_id)
    ]

    
    triplets_list_filtered = []
    num_triplets_filtered = 0
    for triplet in triplets_list:
        print(triplets_list)
        print(len(triplet[0]))
        print(len(triplet[1]))
        if len(triplet[0]) > 0 and len(triplet[1]) > 0:
            triplets_list_filtered.append(triplet)
        else:
            num_triplets_filtered += 1
    
    print(f'Number of triplets filtered: {num_triplets_filtered}')

    if output_file is not None:
        with open(output_file, 'wb') as f:
            pickle.dump(triplets_list_filtered, f)

    return triplets_list_filtered

def is_same_seq(l1, l2):
    if len(l1) != len(l2):
        return False
    s1 = set(tuple(couple) for couple in l1)
    s2 = set(tuple(couple) for couple in l2)
    return s1 == s2

def compare_pkl(pkl_1, pkl_2):
    """
    @brief compare two pkl files
    """
    with open(pkl_1, 'rb') as f:
        pkl_1 = pickle.load(f)
    with open(pkl_2, 'rb') as f:
        pkl_2 = pickle.load(f)

    if len(pkl_1) != len(pkl_2):
        print('Size of pkl files do not match')
        print(f'Pkl 1: {len(pkl_1)} vs Pkl 2: {len(pkl_2)}')
        return False

    # id_to_triplet_1 = {triplet[2]: (triplet[0], triplet[1]) for triplet in pkl_1}
    # for i in tqdm(range(len(pkl_2))):
    #     triplet_2 = pkl_2[i]
    #     if triplet_2[2] not in id_to_triplet_1:
    #         print(f'Triplet {triplet_2[2]} not in pkl 1')
    #         return False
    #     triplet_1 = id_to_triplet_1[triplet_2[2]]
    #     if not is_same_seq(triplet_1[0], triplet_2[0]) or not is_same_seq(triplet_1[1], triplet_2[1]):
    #         print(f'Triplet {triplet_2[2]} entity does not match')
    #         print(f'Types : {triplet_1[0]} vs {triplet_2[0]}')
    #         print(f'KG : {triplet_1[1]} vs {triplet_2[1]}')
    #         return False
    # print('Pkl files are the same')
    # return True

#%%

if __name__ == '__main__':
    graph = 'YAGO43kET'
    for ET_type in ['test', 'train', 'valid']:
        ET_path = f'../data_sampled/ET_{ET_type}_sampled.txt'
        KG_sampled_path = '../data_sampled/KG_train_sampled.txt'
        ET_output_path = f'../data_sampled/LMET_{ET_type}.pkl'

        pkl = convert_to_pkl(f'../data/{graph}', ET_path, ET_output_path, KG_train=KG_sampled_path, ET_train=ET_path)

        print(f'Number of triplets: {len(pkl)}')
        print(compare_pkl(ET_output_path, ET_output_path))
#%%
if __name__ == '__main__':
    graph = 'YAGO43kET'
    for ET_type in ['test', 'valid']:
        ET_path = f'../data/{graph}/ET_{ET_type}.txt'
        KG_sampled_path = f'../data/{graph}/KG_train.txt'
        ET_output_path = f'../data_sampled/LMET_{ET_type}.pkl'


        pkl = convert_to_pkl(f'../data/{graph}', ET_path, ET_output_path, KG_train=KG_sampled_path, ET_train=f'../data/{graph}/ET_train.txt')

        ET_path = f'../data/{graph}/LMET_{ET_type}.pkl'
        print(f'Number of triplets: {len(pkl)}')
        print(compare_pkl(ET_output_path, ET_path))
# %%
