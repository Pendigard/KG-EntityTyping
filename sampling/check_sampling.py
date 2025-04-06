#%%
import os
import pandas as pd
import pickle

def check_sampling(sampling_path):
    df_kg = pd.read_csv(os.path.join(sampling_path, 'KG_train.txt'), header=None, sep='\t', names=['head', 'relation', 'tail'])
    df_ent = pd.read_csv(os.path.join(sampling_path, 'entities.tsv'), sep='\t', header=None, names=['name', 'id'])
    df_type = pd.read_csv(os.path.join(sampling_path, 'types.tsv'), sep='\t', header=None, names=['name', 'id'])
    df_clust = pd.read_csv(os.path.join(sampling_path, 'clusters.tsv'), sep='\t', header=None, names=['name', 'id'])
    df_rel = pd.read_csv(os.path.join(sampling_path, 'relations.tsv'), sep='\t', header=None, names=['name', 'id'])

    if df_kg[df_kg['head'].isin(df_ent['name'])].shape[0] != df_kg.shape[0]:
        print('heads not in entities')
        print(df_kg[~df_kg['head'].isin(df_ent['name'])])
    if df_kg[df_kg['tail'].isin(df_ent['name'])].shape[0] != df_kg.shape[0]:
        print('tails not in entities')
        print(df_kg[~df_kg['tail'].isin(df_ent['name'])])

    df_et = pd.read_csv(os.path.join(sampling_path, 'ET_train.txt'), sep="\t", header=None, names=["entity", "type"])
    df_et = pd.concat([df_et, pd.read_csv(os.path.join(sampling_path, 'ET_valid.txt'), sep="\t", header=None, names=["entity", "type"])], ignore_index=True)
    df_et = pd.concat([df_et, pd.read_csv(os.path.join(sampling_path, 'ET_test.txt'), sep="\t", header=None, names=["entity", "type"])], ignore_index=True)

    if df_et[df_et['entity'].isin(df_ent['name'])].shape[0] != df_et.shape[0]:
        print('entities not in entities')
        print(df_et[~df_et['entity'].isin(df_ent['name'])])

    if df_et[df_et['type'].isin(df_type['name'])].shape[0] != df_et.shape[0]:
        print('types not in types')
        print(df_et[~df_et['type'].isin(df_type['name'])])

    ET_types = ['train', 'valid', 'test']
    for et_type in ET_types:
        with open(os.path.join(sampling_path, f'LMET_{et_type}.pkl'), 'rb') as f:
            LMET = pickle.load(f)
        for et, kg, ent in LMET:
            if ent not in df_ent['id'].values:
                print(f'{et_type} LMET entity not in entities')
                print(ent)
            for triple in kg:
                if triple[0] not in df_ent['id'].values:
                    print(f'{et_type} LMET head not in entities')
                    print(triple[0])
                if triple[2] not in df_ent['id'].values:
                    print(f'{et_type} LMET tail not in entities')
                    print(triple[2])
                if triple[1] >= len(df_rel):
                    if triple[1] - len(df_rel) - len(df_clust) not in df_rel['id'].values:
                        print(f'{et_type} LMET relation not in relations')
                        print(triple[1] - len(df_rel) - len(df_clust))
                else:
                    if triple[1] not in df_rel['id'].values:
                        print(f'{et_type} LMET relation not in relations')
                        print(triple[1])
            for triple in et:
                if triple[0] not in df_ent['id'].values:
                    print(f'{et_type} LMET et entity not in entities')
                    print(triple[0])
                if triple[1] - len(df_rel) not in df_clust['id'].values:
                    print(f'{et_type} LMET cluster not in clusters')
                    print(triple[1] - len(df_rel))
                if triple[2] - len(df_ent) not in df_type['id'].values:
                    print(f'{et_type} LMET type not in types')
                    print(triple[2] - len(df_ent))


check_sampling('../data/YAGO_sampled')
# %%
