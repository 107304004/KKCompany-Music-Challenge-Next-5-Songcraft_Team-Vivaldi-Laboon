from pyserini.search.lucene import LuceneSearcher
from pyserini.search.lucene import querybuilder
from tqdm import tqdm
from util import BM25Searcher


def bm25_search(index_path, output_path, queries, session_items, k):
    searcher = BM25Searcher(index_path)

    # if no set, the default similarity is bm25 with k1=0.9, b=0.4
    searcher.set_bm25()
    
    # prepare the output file
    output = open(output_path, 'w')
    output.write(f'session_id,top1,top2,top3,top4,top5\n')

    for idx, (q_id, q_text) in enumerate(tqdm(queries.items())):
        hits = searcher.search(q_text, k)

        # no repeat recommendation
        if len(set(session_items[idx])) > 17:
            top5 = []
            i = 0
            while (len(top5) < 5) and (i < 30):
                try:
                    if hits[i].docid in session_items[idx]:
                        i += 1
                        continue
                    top5.append(hits[i].docid)
                    i += 1
                except:
                    break
            try:
                output.write(f'{q_id},{top5[0]},{top5[1]},{top5[2]},{top5[3]},{top5[4]}\n')
            except:
                output.write(f'{q_id},{session_items[idx][-1]},{session_items[idx][-2]},{session_items[idx][-3]},{session_items[idx][-4]},{session_items[idx][-5]}\n')
        else:
            try:
                output.write(f'{q_id},{hits[0].docid},{hits[1].docid},{hits[2].docid},{hits[3].docid},{hits[4].docid}\n')
            except:
                output.write(f'{q_id},{session_items[idx][-1]},{session_items[idx][-2]},{session_items[idx][-3]},{session_items[idx][-4]},{session_items[idx][-5]}\n')


def bm25_search_with_score(index_path, output_path, queries, session_items, k):
    searcher = BM25Searcher(index_path)

    # if no set, the default similarity is bm25 with k1=0.9, b=0.4
    searcher.set_bm25()
    
    # prepare the output file
    output = open(output_path, 'w')
    output.write(f'session_id_song_id,bm25_score,bm25_rank\n')

    for idx, (q_id, q_text) in enumerate(tqdm(queries.items())):
        hits = searcher.search(q_text, k)
        rank = 1

        # no repeat recommendation
        if len(set(session_items[idx])) > 17:
            for i in range(k):
                try:
                    if hits[i].docid in session_items[idx]:
                        continue
                    output.write(f'{q_id+"_"+hits[i].docid},{hits[i].score},{rank}\n')
                    rank += 1
                except:
                    break
        else:
            for i in range(k):
                try:
                    output.write(f'{q_id+"_"+hits[i].docid},{hits[i].score},{rank}\n')
                    rank += 1
                except:
                    break


def bm25_search_category(index_path, output_path, queries, session_items, k):
    searcher = BM25Searcher(index_path)

    # if no set, the default similarity is bm25 with k1=0.9, b=0.4
    searcher.set_bm25()

    # prepare the output file
    output = open(output_path, 'w')
    output.write(f'session_id,top1,top2,top3,top4,top5\n')

    for idx, (q_id, q_text) in enumerate(tqdm(queries.items())):
        # reweight important category
        should = querybuilder.JBooleanClauseOccur['should'].value
        boolean_query_builder = querybuilder.get_boolean_query_builder()
        q_text_list = q_text.split(" ")
        # q_text_list = [x for x in q_text_list if x != ""]
        for t in q_text_list:
            try:
                if (q_text_list.count(t) > 1) and ('session' not in t):
                    term = querybuilder.get_term_query(t)
                    boost = querybuilder.get_boost_query(term, 20)
                    boolean_query_builder.add(boost, should)
                else:
                    term = querybuilder.get_term_query(t)
                    boost = querybuilder.get_boost_query(term, 1)
                    boolean_query_builder.add(boost, should)
            except:
                continue

        q_text = boolean_query_builder.build()
        hits = searcher.search(q_text, k)

        # no repeat recommendation
        top5 = []
        for i in range(25):
            try:
                if hits[i].docid in session_items[idx]:
                    i += 1
                    continue
                top5.append(hits[i].docid)
            except:
                break
            if len(top5) == 5:
                break

        while len(top5) < 5:
            top5.append(session_items[idx][-1])

        output.write(f'{q_id},{top5[0]},{top5[1]},{top5[2]},{top5[3]},{top5[4]}\n')
