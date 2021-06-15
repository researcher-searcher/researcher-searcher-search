import json
import pprint
import pandas as pd
from loguru import logger
from workflow.scripts.es_functions import vector_query, standard_query
from workflow.scripts.general import load_spacy_model

vector_index_name = "sentence_vectors"
pp = pprint.PrettyPrinter(indent=4)

test_text1 = (
    "Funding is available from MRC’s Infections and Immunity Board to provide large, "
    "long-term and renewable programme funding for researchers working in the area of "
    "infections and immunity. There is no limit to the funding you can request. This "
    "funding opportunity runs three times every year."
)
test_text2 = (
    "Funding is available from MRC’s Neurosciences and Mental Health Board to support new partnerships between "
    "researchers in the area of neurosciences and mental health. Funding varies widely for partnerships lasting "
    "between one and five years. This funding opportunity runs three times every year."
)
test_text3 = (
    "We have implemented efficient search methods and an application programming interface, to create fast and convenient"
    " functions to utilize triples extracted from the biomedical literature by SemMedDB."
)
test_text4 = (
    "Ankyrin-R provides a key link between band 3 and the spectrin cytoskeleton that helps to maintain the highly "
    "specialised erythrocyte biconcave shape. Ankyrin deficiency results in fragile spherocytic erythrocytes with "
    "reduced band 3 and protein 4.2 expression. We use in vitro differentiation of erythroblasts transduced with shRNAs "
    "targeting the ANK1 gene to generate erythroblasts and reticulocytes with a novel ankyrin-R ‘near null’ human "
    "phenotype with less than 5% of normal ankyrin expression. Using this model we demonstrate that absence of ankyrin "
    "negatively impacts the reticulocyte expression of a variety of proteins including band 3, glycophorin A, spectrin, "
    "adducin and more strikingly protein 4.2, CD44, CD47 and Rh/RhAG. Loss of band 3, which fails to form tetrameric "
    "complexes in the absence of ankyrin, alongside GPA, occurs due to reduced retention within the reticulocyte membrane "
    "during erythroblast enucleation. However, loss of RhAG is temporally and mechanistically distinct, occurring "
    "predominantly as a result of instability at the plasma membrane and lysosomal degradation prior to enucleation. "
    "Loss of Rh/RhAG was identified as common to erythrocytes with naturally occurring ankyrin deficiency and "
    "demonstrated to occur prior to enucleation in cultures of erythroblasts from a hereditary spherocytosis patient "
    "with severe ankyrin deficiency but not in those exhibiting milder reductions in expression. The identification of "
    "prominently reduced surface expression of Rh/RhAG in combination with direct evaluation of ankyrin expression using "
    "flow cytometry provides an efficient and rapid approach for the categorisation of hereditary spherocytosis arising "
    "from ankyrin deficiency."
)
test_text5 = (
    "Risk factors for breast cancer"
)

test_text6 = "neuroscience"

#https://pubmed.ncbi.nlm.nih.gov/25751625/
test_text7 = (
    "Genome-wide association studies (GWAS) and large-scale replication studies have identified common variants in 79 loci associated with breast cancer, explaining ∼14% of the familial risk of the disease. To identify new susceptibility loci, we performed a meta-analysis of 11 GWAS, comprising 15,748 breast cancer cases and 18,084 controls together with 46,785 cases and 42,892 controls from 41 studies genotyped on a 211,155-marker custom array (iCOGS). Analyses were restricted to women of European ancestry. We generated genotypes for more than 11 million SNPs by imputation using the 1000 Genomes Project reference panel, and we identified 15 new loci associated with breast cancer at P < 5 × 10(-8). Combining association analysis with ChIP-seq chromatin binding data in mammary cell lines and ChIA-PET chromatin interaction data from ENCODE, we identified likely target genes in two regions: SETBP1 at 18q12.3 and RNF115 and PDZK1 at 1q21.1. One association appears to be driven by an amino acid substitution encoded in EXO1."
)

# create vectof of each string
def q1(text):
    nlp = load_spacy_model()

    doc = nlp(text)
    for sent in doc.sents:
        logger.info(sent)

        # vectors
        sent_vec = sent.vector
        res = vector_query(index_name=vector_index_name, query_vector=sent_vec)
        if res:
            for r in res[:10]:
                if r["score"] > 0.5:
                    logger.info(f'full sent {r}')

        # noun chunks
        noun_chunk_string = ""
        for chunk in sent.noun_chunks:
            # for token in chunk:
            #    logger.info(token.lemma_)
            # logger.debug(chunk)
            # remove stopwords and things
            if (
                all(token.is_stop != True and token.is_punct != True for token in chunk)
                == True
            ):
                # not sure if should filter on number of words in chunk?
                # might work better here to avoid ambiguous single words, e.g. funding, background...
                if len(chunk) > 0:
                    logger.info(f"noun chunk: {chunk} {len(chunk)}")
                    noun_chunk_string+=str(chunk)+' '
        logger.info(noun_chunk_string)
        if noun_chunk_string != '':
            chunk_vec = nlp(noun_chunk_string).vector
            res = vector_query(index_name=vector_index_name, query_vector=chunk_vec)
            if res:
                for r in res[:10]:
                    if r["score"] > 0.5:
                        logger.info(f'chunk {r}')

# use whole doc as vector
def q2(text):
    nlp = load_spacy_model()
    doc = nlp(text)
    res = vector_query(index_name=vector_index_name, query_vector=doc.vector)
    if res:
        for r in res[0:5]:
            if r["score"] > 0:
                logger.info(pp.pprint(r))
    return res


# standard match against sentence text
def q3(text):
    body={
        # "from":from_val,
        "size": 5,
        "query": {
             "match": {
                "sent_text": {
                    "query": text     
                }
            }
        },
        "_source": ["doc_id","sent_num","sent_text"]
    }
    res = standard_query(index_name=vector_index_name,body=body)
    if res:
        for r in res['hits']['hits']:
            if r["_score"] > 0.5:
                logger.info(pp.pprint(r))
    return res

# combine vectors and full text
def q4(text):
    summary = []
    vector_res = q2(text)
    for r in vector_res[:5]:
        if r["score"] > 0.5:
            summary.append(
                {
                'search':'vector',
                'url':r["url"],
                'sent_num':r['sent_num'],
                'sent_text':r['sent_text'],
                'score':r['score'],
                }
            )
    text_res = q3(text)
    for r in text_res['hits']['hits']:
        #logger.debug(r['_source'])
        summary.append(
            {
            'search':'text',
            'url':r['_source']["doc_id"],
            'sent_num':r['_source']['sent_num'],
            'sent_text':r['_source']['sent_text'],
            'score':r['_score'],
            }
        )
    df = pd.DataFrame(summary)
    print(df)
    print(df['search'].value_counts())
    df.to_csv('workflow/results/search.tsv',sep='\t')

#q1()
#q2()
#q3()
text = 'military health'
#text = 'Text to speech / speech to text'
q1(text)
