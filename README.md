# researcher-searcher-search
Process output data, extract info and index using Elasticsearch 

# .env
```
ELASTIC_VERSION=7.13.2
ELASTIC_SECURITY=true
ELASTIC_PASSWORD=
ELASTIC_HOST=
ELASTIC_HTTP=9200
ELASTIC_HTTPS=9300
ELASTIC_USER=elastic
KIBANA_PORT=5601
LOGSTASH_BEATS_PORT=5044
NAME=name
```

# conda

`conda env create -f environment.yaml`
`conda activate researcher-searcher`

# language models

`python -m spacy download en_core_web_lg`


# Run

## Create NLP data

Run spacy

`snakemake -r parse_text -j1`

Run vector comparisons and tf-idf

`snakemake -r process_text -j1`

Create indexes
- note this creates and starts elasticsearch, kibana and logstash docker containrs

`snakemake -r index_data -j1`

# Issues

Spacy requires memory, especially when parsing abstracts.
- might need to run this in batches if memory low 

# Todo

- Test input files for correct format
