import scispacy
import spacy
import numpy as np
from scipy.spatial import distance
import spacy_universal_sentence_encoder
from loguru import logger


def mark_as_complete(name):
    f = open(name, "w")
    f.write("Done")
    f.close()


def load_spacy_model():
    model_name = "en_core_web_trf"
    model_name = "en_core_web_lg"
    # Load English tokenizer, tagger, parser and NER
    logger.info(f"Loading spacy model {model_name}")
    nlp_web = spacy.load(model_name)
    logger.info(f"Loading spacy model en_use_lg")
    nlp_use = spacy_universal_sentence_encoder.load_model('en_use_lg')
    # nlp = spacy.load("en_core_sci_scibert")
    # nlp = spacy.load("en_core_sci_lg")
    # nlp = spacy.load("en_ner_bionlp13cg_md")
    # nlp.add_pipe("abbreviation_detector")

    # add max length for transformer
    #if model_name == 'en_core_web_trf':
    #    nlp.max_length = 512
    #nlp.max_length=10000
    logger.info("Done...")
    return nlp_web,nlp_use

def create_aaa_distances(vectors=[]):
    logger.info('Creating distances...')
    #https://stackoverflow.com/questions/48838346/how-to-speed-up-computation-of-cosine-similarity-between-set-of-vectors

    logger.info(len(vectors))
    data = np.array(vectors)
    pws = distance.pdist(data, metric='cosine')
    #return as square-form distance matrix
    pws = distance.squareform(pws)
    logger.info(len(pws))
    return pws

#takes an array of vectors
def create_pair_distances(v1=[],v2=[]):
    logger.info('Creating distances...')
    #https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cdist.html#scipy.spatial.distance.cdist

    logger.info(f'{len(v1)} {len(v2)}')
    y = distance.cdist(v1, v2, 'cosine')
    logger.info(len(y))
    return y