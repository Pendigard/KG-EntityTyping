#%%
import pandas as pd
import re
import os


def rel_maker(rel2pass_path: str, relation_list : str='relation_list.csv') -> None:
    """
    This function reads a CSV file containing relations and their corresponding passive forms,
    and creates a dictionary mapping each relation to its passive form.

    :param relation2passive_path: Path to the CSV file containing relations and their passive forms.
    :return: None
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(rel2pass_path, sep='\t', header=None, names=["relation", "passive"])

    df_rel = pd.read_csv(relation_list, sep='\t', header=None, names=["relation"])

    rels = df["relation"].tolist() + df[df['relation'].isin(df_rel['relation'])]['passive'].tolist()
    rels = list(set(rels))  # Remove duplicates
    print(len(rels))

    res = []
    for i, rel in enumerate(rels):
        # Create a dictionary with the relation and its index
        res.append({"relation": rel, "index": i})

    df = pd.DataFrame(res)
    df.to_csv('relations.tsv', sep='\t', index=False, header=False)

def rel2text_maker(rel_path: str) -> None:
    """
    This function reads a CSV file containing relations and their corresponding passive forms,
    and creates a dictionary mapping each relation to its passive form.

    :param relation2passive_path: Path to the CSV file containing relations and their passive forms.
    :return: None
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(rel_path, sep='\t', header=None, names=["relation", "id"])

    def rel2text(rel: str) -> str:
        """
        Convert a relation to its text form by replacing underscores with spaces and capitalizing the first letter.
        """
        spaced_rel = re.sub(r'(?<!^)(?=[A-Z][a-z])', ' ', rel)
        spaced_rel = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', spaced_rel)

        return spaced_rel.lower()
        
    
    df['relation_text'] = df['relation'].apply(rel2text)

    df[['relation', 'relation_text']].to_csv('relation2text.txt', sep='\t', index=False, header=False)

def add_pass_rel(kg_folder_path: str='.', rel2pass_path: str='.', relation_list : str='relation_list.csv') -> None:
    """
    This function reads a CSV file containing relations and their corresponding passive forms,
    and creates a dictionary mapping each relation to its passive form.

    :param relation2passive_path: Path to the CSV file containing relations and their passive forms.
    :return: None
    """
    df_kg = pd.read_csv(os.path.join(kg_folder_path, 'KG_train.txt'), sep='\t', header=None, names=["head", "relation", "tail"])

    df_rel2pass = pd.read_csv(os.path.join(rel2pass_path, 'rel2pass.tsv'), sep='\t', header=None, names=["relation", "passive"])

    df_rel = pd.read_csv(relation_list, sep='\t', header=None, names=["relation"])

    df_kg = df_kg.merge(df_rel2pass, on='relation', how='left')

    df_kg_pass = pd.DataFrame({
        'head': df_kg[df_kg['relation'].isin(df_rel['relation'])]['tail'],
        'relation': df_kg[df_kg['relation'].isin(df_rel['relation'])]['passive'],
        'tail': df_kg[df_kg['relation'].isin(df_rel['relation'])]['head']
    })

    df_kg = pd.concat([df_kg[['head', 'relation', 'tail']], df_kg_pass[['head', 'relation', 'tail']]], ignore_index=True)
    df_kg = df_kg.drop_duplicates()
    df_kg = df_kg.reset_index(drop=True)

    df_kg.to_csv(os.path.join(kg_folder_path, 'KG_train_passive.txt'), sep='\t', index=False, header=False)

if __name__ == "__main__":
    rel_maker("rel2pass.tsv")
    rel2text_maker("relations.tsv")
    add_pass_rel()


# %%

def check_graph(kg_folder_path: str='.'):

    df_rel = pd.read_csv('relations.tsv', sep='\t', header=None, names=["relation", "id"])


    df_kg = pd.read_csv(os.path.join(kg_folder_path, 'KG_train_passive.txt'), sep='\t', header=None, names=["head", "relation", "tail"])
    count_line = df_kg[df_kg['relation'].isin(df_rel['relation'])].shape[0]

    print(df_kg[~df_kg['relation'].isin(df_rel['relation'])])
    return count_line == df_kg.shape[0]


check_graph()
# %%
