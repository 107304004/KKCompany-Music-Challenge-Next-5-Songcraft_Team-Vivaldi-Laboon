import os
import sys
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from pyserini.pyclass import autoclass
from pyserini.search.lucene import LuceneSearcher


class BM25Searcher(LuceneSearcher):

    def __init__(self, index):
        super().__init__(index)

    def set_bm25(self, k1=float(0.9), b=float(0.4)):
        BM25Similarity = autoclass('org.apache.lucene.search.similarities.BM25Similarity')(k1, b)
        self.object.searcher = autoclass('org.apache.lucene.search.IndexSearcher')(self.object.reader)
        self.object.searcher.setMaxClauseCount(65536)
        self.object.searcher.setSimilarity(BM25Similarity)


def read_document(path="data/collection/collection.jsonl"):
    ''' document = item, return {'item_id': features} '''
    data = dict()
    fi = open(path, 'r')
    for line in tqdm(fi):
        item = json.loads(line.strip())
        item_id = item.pop('id')
        features = item.pop('contents')
        data[str(item_id)] = features
    return data


def read_query(doc_path, query_path, l=20):
    ''' query = session, return {'session_id': features} '''
    item_features = read_document(doc_path)

    data = dict()
    sessions = pd.read_parquet(query_path)
    unique_sessions = sessions['session_id'].unique()
    session_items = []
    for session_id in tqdm(unique_sessions):
        features = ""
        items = []
        for item_id in sessions[sessions['session_id'] == session_id]['song_id'][-l:]:
            features = features + item_features[str(item_id)]
        for item_id in sessions[sessions['session_id'] == session_id]['song_id']:
            items.append(item_id)
        # print(features)
        # print(items)
        data[str(session_id)] = features
        session_items.append(items)
    return data, session_items
