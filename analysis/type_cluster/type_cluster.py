# %%
import pandas as pd
import numpy as np
import os
from transformers import BertConfig, BertModel, AutoTokenizer
import seaborn as sns
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import hdbscan
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import json
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cosine
from typing import Literal
import pickle


tqdm.pandas()

#%%

def get_df(graph_path):
	df_kg = pd.read_csv(os.path.join(graph_path, 'KG_train.txt'), sep='\t', header=None, names=['head', 'relation', 'tail'])
	df_et = pd.read_csv(os.path.join(graph_path, 'ET_train.txt'), sep='\t', header=None, names=['entity', 'type'])
	df_et_test = pd.read_csv(os.path.join(graph_path, 'ET_test.txt'), sep='\t', header=None, names=['entity', 'type'])
	df_et_valid = pd.read_csv(os.path.join(graph_path, 'ET_valid.txt'), sep='\t', header=None, names=['entity', 'type'])
	df_type_desc = pd.read_csv(os.path.join(graph_path, 'hier_type_desc.txt'), sep='\t', header=None, names=['type', 'desc'])
	df_et['desc'] = df_et['type'].map(df_type_desc.set_index('type')['desc'])
	df_et_test['desc'] = df_et_test['type'].map(df_type_desc.set_index('type')['desc'])
	df_et_valid['desc'] = df_et_valid['type'].map(df_type_desc.set_index('type')['desc'])

	df_et_group = df_et.groupby('entity').agg({'type': 'count'}).reset_index()
	df_et_group.columns = ['entity', 'type_count']

	df_in = df_kg.groupby('head').agg({'tail': 'count'}).reset_index()
	df_in.columns = ['entity', 'in_degree']
	df_out = df_kg.groupby('tail').agg({'head': 'count'}).reset_index()
	df_out.columns = ['entity', 'out_degree']

	df_degree = pd.merge(df_in, df_out, on='entity', how='outer')
	df_degree = df_degree.fillna(0)
	df_degree['degree'] = df_degree['in_degree'] + df_degree['out_degree']
	df_degree = df_degree.merge(df_et_group, on='entity', how='outer')
	df_degree = df_degree.fillna(0)
	df_degree['full_degree'] = df_degree['degree'] + df_degree['type_count']
	return df_kg, df_et, df_et_test, df_et_valid, df_degree, df_et_group

def remove_prefix(desc):
    if desc.startswith('wikipedia category'):
        desc = desc.replace('wikipedia category', '')
    elif desc.startswith('wordnet'):
        desc = desc.replace('wordnet', '')
    return desc

model_mini_lm = SentenceTransformer('all-MiniLM-L6-v2')

def get_type_embedding(type_str, tokenizer, model):
    inputs = tokenizer(type_str, return_tensors="pt", add_special_tokens=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}  # ✅ Déplacement vers le bon device
    with torch.no_grad():
        outputs = model(**inputs)
    cls_embedding = outputs.last_hidden_state[:, 0, :]  # (1, hidden_size)
    return cls_embedding.squeeze(0).cpu().numpy()  # ✅ CPU pour éviter erreur dans DataFrame

def get_sentence_embedding(sentence):
	sentence = ' '.join(sentence.split()[2:])
	return model_mini_lm.encode(sentence)

def embed_type(df_et_clust, method : Literal['uncased', 'mini_lm', 'uncased_nopre', 'mini_lm_nopre']):

    if method == 'uncased':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        lm_config = BertConfig.from_pretrained('bert-base-uncased')
        lm_encoder = BertModel.from_pretrained('bert-base-uncased', config=lm_config)
        lm_encoder.to(device)
        lm_encoder.eval()
        lm_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

        if 'nopre' in method:
            df_et_clust['type_embedding'] = df_et_clust['desc_no_prefix'].progress_apply(
                lambda x: get_type_embedding(x, lm_tokenizer, lm_encoder)
            )
        else:
            df_et_clust['type_embedding'] = df_et_clust['desc'].progress_apply(
                lambda x: get_type_embedding(x, lm_tokenizer, lm_encoder)
            )
    else:
        if 'nopre' in method:
            df_et_clust['type_embedding'] = df_et_clust['desc_no_prefix'].progress_apply(
                lambda x: get_sentence_embedding(x)
            )
        else:
            df_et_clust['type_embedding'] = df_et_clust['desc'].progress_apply(
                lambda x: get_sentence_embedding(x)
            )
    return df_et_clust

def cluster_and_score(df_ent, embeddings):
	clusterer = hdbscan.HDBSCAN(min_cluster_size=2, metric='euclidean')
	labels = clusterer.fit_predict(embeddings)
	df_ent[f'hdbscan_cluster'] = labels
	df_ent[f'silhouette_score'] = silhouette_score(embeddings, labels) if len(set(labels)) > 1 else -1
	return df_ent

def get_df_cluster(df_et_clust):
    df_clust = pd.DataFrame(columns=['entity', 'type', 'desc', 'set_type', 'type_embedding', 'hdbscan_cluster', 'silhouette_score'])
    for entity in tqdm(df_et_clust['entity'].unique()):
        df_ent = df_et_clust[(df_et_clust['entity'] == entity) & (df_et_clust['set_type'] == 'train')]

        embeddings = np.stack(df_ent['type_embedding'].values)
        df_ent = cluster_and_score(df_ent, embeddings)

        df_clust = pd.concat([df_clust, df_ent], axis=0)
    df_clust = df_clust.reset_index(drop=True)
    return df_clust

def get_df_clust_center(df_clust):

    df_clust_center = df_clust.groupby(['entity', 'hdbscan_cluster']).agg({'type_embedding': 'mean'}).reset_index()
    df_clust_center.columns = ['entity', 'hdbscan_cluster', 'type_embedding_center']
    df_joined = df_clust.merge(
        df_clust_center,
        on=['entity', 'hdbscan_cluster'],
        how='left'
    )

    # 2. Calcul de la distance entre chaque embedding et le centroïde de son cluster
    df_joined['distance_to_center'] = df_joined.apply(
        lambda row: cosine(row['type_embedding'], row['type_embedding_center']),
        axis=1
    )

    # 3. Trouver la distance maximale pour chaque cluster d’une entité
    df_farthest = df_joined.groupby(['entity', 'hdbscan_cluster'])['distance_to_center'].max().reset_index()
    df_farthest.columns = ['entity', 'hdbscan_cluster', 'farthest_distance']

    # 4. Joindre à nouveau si tu veux remettre dans df_clust_center
    df_clust_center = df_clust_center.merge(
        df_farthest,
        on=['entity', 'hdbscan_cluster'],
        how='left'
    )

    return df_clust_center

def get_cluster_threshold(df_clust, df_clust_center):
    # Dictionnaire pour stocker les seuils de distance max par cluster
    cluster_thresholds = {}

    quantile = 0.95  # le seuil de tolérance (ex : 95% des points les plus proches)

    for _, row in df_clust_center.iterrows():
        entity = row['entity']
        cluster = row['hdbscan_cluster']
        center = row['type_embedding_center']

        # On récupère tous les points de ce cluster
        df_cluster_points = df_clust[
            (df_clust['entity'] == entity) &
            (df_clust['hdbscan_cluster'] == cluster) &
            (df_clust['set_type'] == 'train')  # que les points de train
        ]

        # Calcul des distances cosinus au centroïde
        distances = df_cluster_points['type_embedding'].apply(
            lambda x: cosine(x, center)
        )

        if len(distances) == 0:
            continue

        # Seuil basé sur le quantile (ex: 95e percentile)
        threshold = distances.quantile(quantile)
        cluster_thresholds[(entity, cluster)] = threshold
    
    return cluster_thresholds

def get_unsupervised_cluster(df_et_clust, df_clust_center, cluster_thresholds):
    # Résultats stockés ici
    df_clust_unsupervised = pd.DataFrame(columns=[
        'entity', 'type', 'desc', 'set_type',
        'type_embedding',
        'hdbscan_cluster',
        'silhouette_score'
    ])

    for entity in tqdm(df_et_clust['entity'].unique()):
        df_ent = df_et_clust[(df_et_clust['entity'] == entity) & (df_et_clust['set_type'] != 'train')]
        df_clust_ent = df_clust_center[(df_clust_center['entity'] == entity) & (df_clust_center['hdbscan_cluster'] != -1)]

        if df_clust_ent.empty or df_ent.empty:
            continue

        for _, row in df_ent.iterrows():
            embedding = row['type_embedding']

            best_cluster = -1
            best_distance = np.inf

            # Parcours des clusters connus de l'entité
            for _, clust_row in df_clust_ent.iterrows():
                cluster_id = clust_row['hdbscan_cluster']
                center = clust_row['type_embedding_center']

                dist = cosine(embedding, center)

                threshold = cluster_thresholds.get((entity, cluster_id), None)

                if threshold is not None and dist <= threshold and dist < best_distance:
                    best_distance = dist
                    best_cluster = cluster_id

            # Ajout du résultat
            df_clust_unsupervised.loc[len(df_clust_unsupervised)] = {
                'entity': row['entity'],
                'type': row['type'],
                'desc': row['desc'],
                'set_type': row['set_type'],
                'type_embedding': embedding,
                'hdbscan_cluster': best_cluster,
                'silhouette_score': None
            }
    return df_clust_unsupervised
    

def clusterize(graph_path, high_type_threshold=10, method='uncased'):
    df_kg, df_et, df_et_test, df_et_valid, df_degree, df_et_group = get_df(graph_path)
    high_type_entity = df_degree[df_degree['type_count'] > high_type_threshold]['entity'].tolist()

    df_et_clust = df_et[df_et['entity'].isin(high_type_entity)]
    df_et_clust['set_type'] = 'train'

    df_et_test_clust = df_et_test[df_et_test['entity'].isin(high_type_entity)]
    df_et_test_clust['set_type'] = 'test'
    df_et_clust = pd.concat([df_et_clust, df_et_test_clust], axis=0)

    df_et_valid_clust = df_et_valid[df_et_valid['entity'].isin(high_type_entity)]
    df_et_valid_clust['set_type'] = 'valid'
    df_et_clust = pd.concat([df_et_clust, df_et_valid_clust], axis=0)

    df_et_clust['desc_no_prefix'] = df_et_clust['desc'].apply(lambda x: remove_prefix(x))
    df_et_clust = embed_type(df_et_clust, method)

    with open('df_et_clust.pkl', 'wb') as f:
        df_et_clust.to_pickle(f)

    df_clust = get_df_cluster(df_et_clust)

    print('Average silhouette score:', df_clust['silhouette_score'].mean())

    df_clust_center = get_df_clust_center(df_clust)
    cluster_threshold = get_cluster_threshold(df_clust, df_clust_center)
    df_clust_unsupervised = get_unsupervised_cluster(df_et_clust, df_clust_center, cluster_threshold)
    return df_clust_unsupervised, df_clust

def generate_files(df_clust_unsupervised, df_clust, output_path):
    et_train = pd.read_csv('../../data/YAGO_sampled_valid/ET_train.txt', sep='\t', header=None, names=['entity', 'type'])
    et_test = pd.read_csv('../../data/YAGO_sampled_valid/ET_test.txt', sep='\t', header=None, names=['entity', 'type'])
    et_valid = pd.read_csv('../../data/YAGO_sampled_valid/ET_valid.txt', sep='\t', header=None, names=['entity', 'type'])
    entities = pd.read_csv('../../data/YAGO_sampled_valid/entities.tsv', sep='\t', header=None, names=['entity', 'entity_id'])
    last_id = entities['entity_id'].max() + 1
    entity_wiki = json.load(open('../../data/YAGO_sampled_valid/entity_wiki.json', 'r', encoding='utf-8'))
    kg_train = pd.read_csv('../../data/YAGO_sampled_valid/KG_train.txt', sep='\t', header=None, names=['head', 'relation', 'tail'])
    kg_train.to_csv(os.path.join(output_path,'KG_train.txt'), index=False, sep='\t', header=False)

    for i, row in tqdm(df_clust.iterrows()):
        if row['hdbscan_cluster'] == -1:
            continue
        entity_name = row['entity'] + '_' + str(int(row['hdbscan_cluster']))
        if entity_name not in entities['entity'].values:
            entities.loc[len(entities)] = [entity_name, last_id]
            last_id += 1
            entity_wiki[entity_name] = entity_wiki[row['entity']]
            kg_ent = kg_train[(kg_train['head'] == row['entity']) | (kg_train['tail'] == row['entity'])].copy()
            kg_ent['head'] = kg_ent['head'].replace(row['entity'], entity_name)
            kg_ent['tail'] = kg_ent['tail'].replace(row['entity'], entity_name)
            kg_ent[['head', 'relation', 'tail']].to_csv(os.path.join(output_path,'KG_train.txt'), index=False, sep='\t', header=False, mode='a')
            
        et_tmp = pd.DataFrame([{'entity' : entity_name, 'type' : row['type']}])
        et_train = pd.concat([et_train, et_tmp], axis=0)

    for i, row in df_clust_unsupervised.iterrows():
        if row['hdbscan_cluster'] == -1:
            continue
        entity_name = row['entity'] + '_' + str(row['hdbscan_cluster'])
        if entity_name not in entities['entity'].values:
            print('Adding entity:', entity_name)
            entities.loc[len(entities)] = [entity_name, last_id]
            last_id += 1
            entity_wiki[entity_name] = entity_wiki[row['entity']]
            kg_ent = kg_train[(kg_train['head'] == row['entity']) | (kg_train['tail'] == row['entity'])].copy()
            kg_ent['head'] = kg_ent['head'].replace(row['entity'], entity_name)
            kg_ent['tail'] = kg_ent['tail'].replace(row['entity'], entity_name)
            kg_ent[['head', 'relation', 'tail']].to_csv(os.path.join(output_path,'KG_train.txt'), index=False, sep='\t', header=False, mode='a')

        if row['set_type'] == 'test':
            et_test.loc[(et_test['entity'] == row['entity']) & (et_test['type'] == row['type']), 'entity'] = entity_name
        else:
            et_valid.loc[et_valid['entity'] == row['entity'], 'entity'] = entity_name

    entities.to_csv(os.path.join(output_path, 'entities.tsv'), sep='\t', index=False, header=False)
    et_train.to_csv(os.path.join(output_path, 'ET_train.txt'), sep='\t', index=False, header=False)
    et_test.to_csv(os.path.join(output_path, 'ET_test.txt'), sep='\t', index=False, header=False)
    et_valid.to_csv(os.path.join(output_path, 'ET_valid.txt'), sep='\t', index=False, header=False)
    with open(os.path.join(output_path, 'entity_wiki.json'), 'w', encoding='utf-8') as f:
        json.dump(entity_wiki, f, ensure_ascii=False, indent=4)
# %%

df_clust_unsupervised, df_clust = clusterize('../../data/YAGO_sampled_valid/', method='mini_lm_nopre')

# %%

output_path = 'files2'
if not os.path.exists(output_path):
    os.makedirs(output_path)
generate_files(df_clust_unsupervised, df_clust, output_path)

# %%


no_noise_entity = df_clust.groupby('entity').agg({'hdbscan_cluster': lambda x: True if -1 in x.values else False}).reset_index()

no_noise_entity = no_noise_entity[no_noise_entity['hdbscan_cluster'] == False]['entity'].tolist()

# %%

one_clust_test = df_clust_unsupervised[df_clust_unsupervised['set_type'] == 'test'].groupby('entity').agg({'hdbscan_cluster': lambda x: True if len(set(x)) > 1 else False}).reset_index()
one_clust_test = one_clust_test[one_clust_test['hdbscan_cluster'] == True]['entity'].tolist()


# %%

with open('df_clust_unsupervised.pkl', 'wb') as f:
    df_clust_unsupervised.to_pickle(f)
with open('df_clust.pkl', 'wb') as f:
    df_clust.to_pickle(f)
# %%

with open('df_clust_unsupervised.pkl', 'rb') as f:
    df_clust_unsupervised = pd.read_pickle(f)
with open('df_clust.pkl', 'rb') as f:
    df_clust = pd.read_pickle(f)

# %%

df_clust_unsupervised

# %%
