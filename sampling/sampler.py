#%%
import os
import pandas as pd
import shutil

from KG_graph_sampler import KG_sampling
from filter_ET import filter_ET
from KG_pkl_converter import convert_to_pkl

def make_entities(output_folder):
    """
    Create a mapping of entities to indices and save it to a file.

    """
    df_kg = pd.read_csv(os.path.join(output_folder, 'KG_train.txt'), sep="\t", header=None, names=["head", "relation", "tail"])
    entities = set(df_kg["head"]).union(set(df_kg["tail"]))
    entity_to_index = {entity: i for i, entity in enumerate(entities)}
    
    with open(os.path.join(output_folder, 'entities.tsv'), 'w', encoding='utf-8') as f:
        for entity, index in entity_to_index.items():
            f.write(f"{entity}\t{index}\n")

def make_ET(graph_folder, output_folder):
    ET_types = ['test', 'train', 'valid']
    for ET_type in ET_types:
        ET_file = os.path.join(graph_folder, f'ET_{ET_type}.txt')
        KG_sampled_file = os.path.join(output_folder, 'KG_train.txt')
        ET_output_file = os.path.join(output_folder, f'ET_{ET_type}.txt')
        print(f'Filtering ET file: {ET_file}')
        filter_ET(ET_file, KG_sampled_file, ET_output_file)

def make_ET_types(output_folder):
    """
    Create ET types files and convert them to pkl format.
    
    Args:
        output_folder (str): The path to the folder where the ET types will be saved.
    """
    df_et = pd.read_csv(os.path.join(output_folder, 'ET_train.txt'), sep="\t", header=None, names=["entity", "type"])
    df_et = pd.concat([df_et, pd.read_csv(os.path.join(output_folder, 'ET_valid.txt'), sep="\t", header=None, names=["entity", "type"])], ignore_index=True)
    df_et = pd.concat([df_et, pd.read_csv(os.path.join(output_folder, 'ET_test.txt'), sep="\t", header=None, names=["entity", "type"])], ignore_index=True)

    types = set(df_et["type"])
    type_to_index = {type_: i for i, type_ in enumerate(types)}

    with open(os.path.join(output_folder, 'types.tsv'), 'w', encoding='utf-8') as f:
        for type_, index in type_to_index.items():
            f.write(f"{type_}\t{index}\n")

def make_pkl(output_folder, banned_ids=None, valid=True):
    ET_types = ['test', 'train', 'valid']
    for ET_type in ET_types:
        ET_file = os.path.join(output_folder, f'ET_{ET_type}.txt')
        pkl_output_file = os.path.join(output_folder, f'LMET_{ET_type}.pkl')
        KG_sampled_file = os.path.join(output_folder, 'KG_train.txt')
        ET_output_file = os.path.join(output_folder, f'ET_{ET_type}.txt')
        print(f'Converting ET file to pkl: {ET_file}')
        convert_to_pkl(output_folder, KG_sampled_file, ET_output_file, pkl_output_file, banned_ids=banned_ids, valid=valid)

def sample_graph(graph_folder, output_folder):
    """
    Sample a knowledge graph from the given folder and save it to the output folder.
    
    Args:
        graph_folder (str): The path to the folder containing the knowledge graph files.
        output_folder (str): The path to the folder where the sampled graph will be saved.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    KG_sampling(graph_folder, output_folder=output_folder)
    # We copy the files that don't change
    # in YAGO there are too few relations to sample them so we copy them all
    shutil.copy(os.path.join(graph_folder, 'relations.tsv'), os.path.join(output_folder, 'relations.tsv'))
    # Clusters are not used in the SEM so we don't need to sample them
    shutil.copy(os.path.join(graph_folder, 'clusters.tsv'), os.path.join(output_folder, 'clusters.tsv'))

    # We keep all the file that gives the text version of each element in the graph
    shutil.copy(os.path.join(graph_folder, 'relation2text.txt'), os.path.join(output_folder, 'relation2text.txt'))
    shutil.copy(os.path.join(graph_folder, 'entity_wiki.json'), os.path.join(output_folder, 'entity_wiki.json'))
    shutil.copy(os.path.join(graph_folder, 'hier_type_desc.txt'), os.path.join(output_folder, 'hier_type_desc.txt'))

    make_entities(graph_folder, output_folder)
    make_ET(graph_folder, output_folder)
    make_ET_types(output_folder)
    make_pkl(output_folder)

    
#%%
sample_graph("../data/YAGO43kET", "../data/YAGO_sampled")

# %%


make_pkl("../data/YAGO_sampled_mirror_true", valid=True)

# %%

make_entities("../data/YAGO_sampled_passive")

# %%
