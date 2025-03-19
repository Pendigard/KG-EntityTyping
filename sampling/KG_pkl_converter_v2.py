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


def convert_to_pkl(KG_path, KG_file, ET_file, output_file, get_clust_name_fonc=get_cluster_name_from_type):
    """
    @brief convert a knowledge graph into a pkl accepted by SSET
    @param KG_path : folder where clusters.tsv, entities.tsv and type.tsv are present
    @param KG_file : KG file to convert (Formatted as -> entity_A\trel\tentity_B)
    @param ET_file : ET file to convert (Formatted as -> entity\ttype)
    @param get_clust_name_fonc : function to get cluster from type must take type and df_clust as input
    @param output_file : pkl generated
    """
    rel_path = os.path.join(KG_path, 'relations.tsv')
    df_rel = pd.read_csv(rel_path, sep='\t', header=None, names=['relation', 'id'])
    rel_count = len(df_rel)

    ent_path = os.path.join(KG_path, 'entities.tsv')
    df_ent = pd.read_csv(ent_path, sep='\t', header=None, names=['entity', 'id'])
    ent_count = len(df_ent)

    type_path = os.path.join(KG_path, 'types.tsv')
    df_type = pd.read_csv(type_path, sep='\t', header=None, names=['type', 'id'])
    df_type['id'] += ent_count

    clust_path = os.path.join(KG_path, 'clusters.tsv')
    df_clust = pd.read_csv(clust_path, sep='\t', header=None, names=['cluster', 'id'], keep_default_na=False)
    df_clust['id'] += rel_count
    
    df_kg = pd.read_csv(KG_file, sep='\t', header=None, names=['entity', 'relation', 'entity_2'])
    
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
    df_et = pd.read_csv(ET_file, sep='\t', header=None, names=['entity', 'type'])

    df_et = df_et[df_et['type'].isin(set(df_type['type'].values))]

    df_type = df_type[df_type['type'].isin(set(df_et['type'].values))]

    df_ent = df_ent[df_ent['entity'].isin(set(df_et['entity'].values))]
    
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
    for ent_id in tqdm(df_ent['id'])
    ]

    
    triplets_list_filtered = []
    num_triplets_filtered = 0
    for triplet in triplets_list:
        if len(triplet[0]) > 0 and len(triplet[1]) > 0:
            triplets_list_filtered.append(triplet)
        else:
            num_triplets_filtered += 1
    
    print(f'Number of triplets filtered: {num_triplets_filtered}')

    if output_file is not None:
        with open(output_file, 'wb') as f:
            pickle.dump(triplets_list_filtered, f)

    return triplets_list_filtered, num_triplets_filtered

if __name__ == '__main__':
    graph = 'YAGO43kET'
    for ET_type in ['test', 'train', 'valid']:
        ET_path = f'../data_sampled/ET_{ET_type}_sampled.txt'
        KG_sampled_path = '../data_sampled/KG_train_sampled.txt'
        ET_output_path = f'../data_sampled/LMET_{ET_type}.pkl'

        pkl, num = convert_to_pkl(f'../data/{graph}', KG_sampled_path, ET_path, ET_output_path)

        len_pkl = len(pkl)
        len_ET = pd.read_csv(f'../data_sampled/ET_{ET_type}_sampled.txt', sep='\t', header=None, names=['entity', 'type'])['entity'].nunique()
        if len_pkl + num != len_ET:
            raise ValueError(f'Size of pkl and ET file do not match for {ET_type} set : pkl_size {len_pkl} + {num} flitered != {len_ET}')