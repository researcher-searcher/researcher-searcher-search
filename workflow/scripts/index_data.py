import pandas as pd
from simple_parsing import ArgumentParser
from loguru import logger
from workflow.scripts.general import mark_as_complete, load_spacy_model
from workflow.scripts.es_functions import (
    create_vector_index,
    create_mean_vector_index,
    create_noun_index,
    delete_index,
    index_vector_data,
    index_mean_vector_data,
    index_noun_data,
    boost_index
)

dim_size=512

parser = ArgumentParser()

parser.add_argument("--input", type=str, help="Input file prefix")
parser.add_argument("--output", type=str, help="Output file prefix")
args = parser.parse_args()

vector_outfile = "workflow/results/text_data_vectors.pkl.gz"
noun_outfile = "workflow/results/text_data_noun_chunks.tsv.gz"
people_vector = "workflow/results/people_vectors.pkl.gz"
research_vector = "workflow/results/research_vectors.pkl.gz"

def index_vectors():
    vector_index_name = "use_title_sentence_vectors"
    delete_index(vector_index_name)
    create_vector_index(index_name=vector_index_name, dim_size=dim_size)
    df = pd.read_pickle(vector_outfile)
    index_vector_data(
        df=df, index_name=vector_index_name, text_type='title'
    )
    vector_index_name = "use_abstract_sentence_vectors"
    delete_index(vector_index_name)
    create_vector_index(index_name=vector_index_name, dim_size=dim_size)
    df = pd.read_pickle(vector_outfile)
    index_vector_data(
        df=df, index_name=vector_index_name, text_type='abstract'
    )

def index_mean_vectors():
    vector_index_name = "use_person_vectors"
    delete_index(vector_index_name)
    create_mean_vector_index(index_name=vector_index_name, dim_size=dim_size)
    df = pd.read_pickle(people_vector)
    index_mean_vector_data(
        df=df, index_name=vector_index_name, id_field='email'
    )
    vector_index_name = "use_output_vectors"
    delete_index(vector_index_name)
    create_mean_vector_index(index_name=vector_index_name, dim_size=dim_size)
    df = pd.read_pickle(research_vector)
    index_mean_vector_data(
        df=df, index_name=vector_index_name, id_field='url'
    )

def index_nouns():
    noun_index_name = "use_title_sentence_nouns"
    delete_index(noun_index_name)
    create_noun_index(index_name=noun_index_name)
    df = pd.read_csv(noun_outfile,sep='\t')
    index_noun_data(
        df=df,index_name=noun_index_name, text_type='title'
    )

    noun_index_name = "use_abstract_sentence_nouns"
    delete_index(noun_index_name)
    create_noun_index(index_name=noun_index_name)
    df = pd.read_csv(noun_outfile,sep='\t')
    index_noun_data(
        df=df,index_name=noun_index_name, text_type='abstract'
    )

if __name__ == "__main__":
    index_vectors()
    index_nouns()
    index_mean_vectors()
    mark_as_complete(args.output)
