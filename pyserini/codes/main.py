import argparse
from util import read_query
from search import bm25_search, bm25_search_category, bm25_search_with_score


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", default="indexes/song_collection", type=str)
    parser.add_argument("--doc_path", default="data/song_collection/song_collection_jsonl", type=str)
    parser.add_argument("--query_path", default="../data/label_test_source.parquet", type=str)
    parser.add_argument("--k", default=50, type=int)
    parser.add_argument("--last", default=20, type=int)
    parser.add_argument("--output_path", default="../subs/bm25.csv")
    
    args = parser.parse_args()

    queries, session_items = read_query(args.doc_path, args.query_path, l=args.last)

    # result stored in output_path
    # bm25_search(args.index, args.output_path, queries, session_items, args.k)
    bm25_search_category(args.index, args.output_path, queries, session_items, args.k)
    # bm25_search_with_score(args.index, args.output_path, queries, session_items, args.k)
