# -*- coding: utf-8 -*-
import pickle
import pandas as pd
import os
from tqdm import tqdm


#ce que vous donner:
#KG_path: global ou vous avez les kg,et,relations.txt...
#KG_file: nom de votre kg file ,par ex: kg_train.txt
#ET_file: nom de et file que vous voulez le combiner avec kg_train, par ex: et_train/test/valid...abs
#output_file: nom vous voulez pour le lmet pkl en output par ex: lmet_train_sampled...

#trouver le nom de cluster, pas important cest utiliser dans la fonction apres  apres
def get_cluster_name_from_type(type_name):
    if len(type_name.split('_')) < 2:
        return type_name
    return type_name.split('_')[1]

#encapsulation: vous appelez directement cette fonction et lui donne touts les parametres et il fait tout.
def convert_to_pkl(KG_path, KG_file, ET_file, output_file, get_clust_name_fonc = get_cluster_name_from_type):
    tqdm.pandas()

    # loading......
    df_rel = pd.read_csv(os.path.join(KG_path, 'relations.tsv'), sep='\t', header=None, names=['relation', 'id'])
    df_ent = pd.read_csv(os.path.join(KG_path, 'entities.tsv'), sep='\t', header=None, names=['entity', 'id'])
    df_type = pd.read_csv(os.path.join(KG_path, 'types.tsv'), sep='\t', header=None, names=['type', 'id'])
    df_clust = pd.read_csv(os.path.join(KG_path, 'clusters.tsv'), sep='\t', header=None, names=['cluster', 'id'], keep_default_na=False)

    # ajoutons le totalnbrelations et totalnbentity pour faire unicite 
    df_type['id'] += len(df_ent)
    df_clust['id'] += len(df_rel)

    # loading les kg et et
    df_kg = pd.read_csv(KG_file, sep='\t', header=None, names=['entity', 'relation', 'entity_2'])
    df_et = pd.read_csv(ET_file, sep='\t', header=None, names=['entity', 'type'])
    df_et_train = pd.read_csv(os.path.join(KG_path, 'ET_train.txt'), sep='\t', header=None, names=['entity', 'type'])
    df_et_valid = pd.read_csv(os.path.join(KG_path, 'ET_valid.txt'), sep='\t', header=None, names=['entity', 'type'])

    df_et_train_full = pd.concat([df_et_train, df_et_valid], ignore_index=True)
    # Le pkl original de LMET est construit sur le train et le valid, donc on fait pareil ici

    # select les entities dans kg
    kg_entities = set(df_kg['entity']).union(set(df_kg['entity_2']))
    df_ent_kg = df_ent[df_ent['entity'].isin(kg_entities)]

    # select les entities dans et
    # df_et = df_et[df_et['type'].isin(df_type['type'])]
    df_ent_et = df_ent[df_ent['entity'].isin(df_et['entity'])]

    # merge!
    df_ent_final = df_ent_kg[df_ent_kg['entity'].isin(df_ent_et['entity'])]

    # pour KG constuire sa moitie de structure dans LMET pkl final
    df_kg_id = (
        df_kg.merge(df_ent_kg, on="entity")
            .rename(columns={"id": "id_1", "entity": "entity_1", "entity_2": "entity"})
            .merge(df_ent_kg, on="entity")
            .rename(columns={"id": "id_2", "entity": "entity_2"})
            .merge(df_rel, on="relation")
            .rename(columns={"id": "rel_id"})
            [["id_1", "rel_id", "id_2"]]
    )

    # ici une petite amelioration mais pas tres important
    def safe_get_cluster_id(t):
        cluster = get_clust_name_fonc(t)
        match = df_clust[df_clust['cluster'] == cluster]
        if match.empty:
            return None
        return match['id'].iloc[0]

    df_type['clust_id'] = df_type['type'].progress_apply(safe_get_cluster_id)
    df_type = df_type[df_type['clust_id'].notnull()]

    # pour ET constuire sa moitie de structure dans LMET pkl final
    df_et_id = (
        df_et_train_full.merge(df_type, on='type')[['entity', 'type', 'id', 'clust_id']]
            .rename(columns={'id': 'type_id'})
            .merge(df_ent, on='entity')
            .rename(columns={'id': 'ent_id'})[['ent_id', 'clust_id', 'type_id']]
    )

    # moitie KG + moitie ET 
    triplets_list = []
    for eid in tqdm(df_ent_final['id']):
        et = df_et_id[df_et_id['ent_id'] == eid].values.tolist()
        kg = df_kg_id[(df_kg_id['id_1'] == eid) | (df_kg_id['id_2'] == eid)].values.tolist()
        if len(et) == 0: # Cas où il n'y a pas de type pour l'entité en train
            # On ajoute le dernier type (Comme dans le LMET original)
            last_type = df_type['id'].iloc[-1]
            last_clust = df_type['clust_id'].iloc[-1] # Prend le cluster du dernier type
            et.append([eid, last_clust, last_type]) 
        for i in range(len(kg)):
            kg_rel = kg[i]
            if kg_rel[2] == eid: # Cas où la relation est entrante, ent1 rel ent2 -> ent2 rel+num_cluster+num_rel ent1 (Comme dans le LMET original)
                kg_rel = kg_rel[::-1]
                kg_rel[1] = kg_rel[1] + len(df_clust) + len(df_rel)
            kg[i] = kg_rel
        triplets_list.append((et, kg, eid))
        
    # sauvegarder comme pkl, le nom est donne par vous dans la parametre output_file
    if output_file is not None:
        with open(output_file, 'wb') as f:
            pickle.dump(triplets_list, f)

    return triplets_list


def compare_pkl_files(file1, file2):
    with open(file1, 'rb') as f:
        data1 = pickle.load(f)
    with open(file2, 'rb') as f:
        data2 = pickle.load(f)

    if len(data1) != len(data2):
        print(f"Files have different lengths: {len(data1)} vs {len(data2)}")
        return

    dico_pkl_1 = {}

    for triplet in data1:
        dico_pkl_1[triplet[2]] = (triplet[0], triplet[1])

    for triplet in data2:
        if triplet[2] not in dico_pkl_1:
            print(f"triplet non sample: {triplet}")
            continue
        et, kg = dico_pkl_1[triplet[2]]
        if len(et) != len(triplet[0]):
            print(f'The two pkl have different ET sizes for entity {triplet[2]}')
            print(f'pkl1_ET: {et}')
            print(f'pkl2_ET: {triplet[0]}')
        if len(kg) != len(triplet[1]):
            print(f'The two pkl have different KG sizes for entity {triplet[2]}')
            print(f'pkl1_KG: {kg}')
            print(f'pkl2_KG: {triplet[1]}')
        for et_triplet in triplet[0]:
            if et_triplet not in et:
                print(f'The two pkl have different ET values for entity {triplet[2]}')
                print(f'pkl1_ET: {et}')
                print(f'pkl2_ET: {triplet[0]}')
        for kg_triplet in triplet[1]:
            if kg_triplet not in kg:
                print(f'{kg_triplet} not in kg')
                print(f'The two pkl have different KG values for entity {triplet[2]}')
                print(f'pkl1_KG: {kg}')
                print(f'pkl2_KG: {triplet[1]}')
    # If no differences were found
    print(f"Files {file1} and {file2} are identical.")



if __name__ == "__main__":
    # Exemple d'utilisation
    kg_set = 'train'
    KG_path = '../data/YAGO43kET' # chemin vers le dossier contenant les fichiers KG non échantillonnés
    # Les fichiers ET non échantillonné et les .tsv
    KG_file = '../data/YAGO43kET/KG_train.txt' # KG samplé toujours train
    ET_file = f'../data/YAGO43kET/ET_{kg_set}.txt' # L'ET samplé
    output_file = f'lmet_{kg_set}_sampled.pkl'
    triplets_list = convert_to_pkl(KG_path, KG_file, ET_file, output_file, get_clust_name_fonc=get_cluster_name_from_type)
    
#    compare_pkl_files(f'../data/YAGO43kET/LMET_{kg_set}.pkl', f'lmet_test_{kg_set}.pkl') # Fonction de comparaison

