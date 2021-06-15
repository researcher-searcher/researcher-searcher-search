import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from loguru import logger

corpus = pd.read_csv("workflow/results/text_data_noun_chunks.tsv.gz",sep='\t')
logger.info(corpus.head())
#logger.debug(corpus['noun_phrase'])

# http://www.davidsbatista.net/blog/2018/02/28/TfidfVectorizer/
def dummy_fun(doc):
    return doc

def vectorize_corpus(corpus):
    vectorizer = TfidfVectorizer(
        analyzer = 'word',
        tokenizer=dummy_fun,
        preprocessor=dummy_fun,
        token_pattern=None
        )
    # create list of lists
    corpus_data = []
    for i in list(corpus['noun_phrase'].str.lower()):
        corpus_data.append([i])
    vectorizer.fit_transform(corpus_data)
    #logger.info(vectorizer.get_feature_names())
    return vectorizer

def tfidf_doc(tfidf='',text=[]):
    #transform function transforms a document to document-term matrix
    response = tfidf.transform([text])

    #get the feature name from the model
    feature_names = tfidf.get_feature_names()
    res={}
    sorted_res = []
    for col in response.nonzero()[1]:
        res[feature_names[col]]=response[0, col]
        #reverse sort the results
        sorted_res = sorted(res.items(), key=lambda kv: kv[1], reverse=True)
    return sorted_res

def person_to_output():
    logger.info('Getting person to output data...')
    rm_df = pd.read_csv("workflow/results/research_metadata.tsv.gz",sep='\t')
    logger.info(rm_df.shape)
    noun_df = pd.read_csv("workflow/results/text_data_noun_chunks.tsv.gz",sep='\t')
    logger.info(noun_df.shape)
    m = rm_df.merge(noun_df,left_on="url",right_on="url")[['email','noun_phrase']]
    m['noun_phrase'] = m['noun_phrase'].str.lower()
    logger.info(m.shape)
    logger.info(m.head())
    grouped = m.groupby('email')['noun_phrase'].apply(list).reset_index(name = 'noun_list')
    return grouped

def run():
    vectorizer = vectorize_corpus(corpus)
    feature_names = vectorizer.get_feature_names()
    df = pd.DataFrame(feature_names)
    logger.info(df.head())
    df.to_csv("workflow/results/noun_chunks.tsv.gz",index=False,header=['noun_chunk'])
    person_noun_chunks = person_to_output()
    logger.info(person_noun_chunks)
    res = []
    for i,row in person_noun_chunks.iterrows():
        logger.info(f"{i} {row['email']}")
        sorted_res = tfidf_doc(tfidf=vectorizer,text=row['noun_list'])
        #logger.info(sorted_res)
        for i in sorted_res[:100]:
            res.append({'email':row['email'],'noun_chunk':i[0],'score':i[1]})
    #logger.info(res)
    df = pd.DataFrame(res)
    df.to_csv("workflow/results/noun_chunks_tfidf.tsv.gz",sep='\t',index=False)

if __name__ == "__main__":
    run()
