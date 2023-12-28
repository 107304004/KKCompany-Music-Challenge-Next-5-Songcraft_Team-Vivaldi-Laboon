import nltk
from nltk.util import pad_sequence
from nltk import ngrams
from nltk.lm import MLE
from nltk.lm.preprocessing import padded_everygram_pipeline

import pandas as pd
import numpy as np
import random
from tqdm import tqdm
from tqdm.contrib import tzip
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--n", "-n", default=4, type=int)
parser.add_argument("--output_path", "-o", type=str)
args = parser.parse_args()


train_source = pd.read_parquet("../data/label_train_source.parquet")
train_target = pd.read_parquet("../data/label_train_target.parquet")
test_source = pd.read_parquet("../data/label_test_source.parquet")
df = pd.concat([train_source, train_target, test_source], axis=0).sort_values(["session_id", "listening_order"])[["session_id", "song_id"]]
corpus = df.groupby('session_id')['song_id'].apply(list).tolist()

# Padded corpus
padded_corpus = [list(pad_sequence(d, 
                                   pad_left=True, left_pad_symbol="<s>", 
                                   pad_right=True, right_pad_symbol="</s>", 
                                   n = args.n)) for d in corpus]


print("Building 4-gram...")
# Pad the n-grams and create training data
train_data, padded_vocab = padded_everygram_pipeline(args.n, padded_corpus)
model = MLE(args.n)
model.fit(train_data, padded_vocab)


print("Generate song_id")
def generate_score(sessions, corpus, args, output_path):
    output = open(output_path, 'w')
    output.write(f'session_id_song_id,ngram_score,ngram_rank\n')

    for s_id, doc in tzip(sessions, corpus):
        rank = 1

        # next 5 words by song 18~20
        doc_18_20 = doc[-(args.n - 1):]
        next_5_words_18_20 = []
        while len(next_5_words_18_20) < 100:
            # 4gram
            next_word, count = model.generate(1, doc_18_20[-(args.n - 1):])
            # 3gram
            if next_word == '</s>':
                next_word, count = model.generate(1, doc_18_20[-(args.n - 2):])
            # break if end of sentence
            if next_word == '</s>':
                break

            output.write(f'{str(s_id)+"_"+next_word},{count},{rank}\n')
            rank += 1
            next_5_words_18_20.append(next_word)
            doc_18_20 += [next_word]

        # next 5 words by song 17~19
        doc_17_19 = doc[-(args.n - 0):-1]
        next_5_words_17_19 = []
        while len(next_5_words_17_19) < 100:
            # 4gram
            next_word, count = model.generate(1, doc_17_19[-(args.n - 1):])
            # 3gram
            if next_word == '</s>':
                next_word, count = model.generate(1, doc_17_19[-(args.n - 2):])
            # break if end of sentence
            if next_word == '</s>':
                break

            output.write(f'{str(s_id)+"_"+next_word},{count},{rank}\n')
            rank += 1
            next_5_words_17_19.append(next_word)
            doc_17_19 += [next_word]

        # next 5 words by song 16~18
        doc_16_18 = doc[-(args.n + 1):-2]
        next_5_words_16_18 = []
        while len(next_5_words_16_18) < 100:
            # 4gram
            next_word, count = model.generate(1, doc_16_18[-(args.n - 1):])
            # 3gram
            if next_word == '</s>':
                next_word, count = model.generate(1, doc_16_18[-(args.n - 2):])
            # break if end of sentence
            if next_word == '</s>':
                break

            output.write(f'{str(s_id)+"_"+next_word},{count},{rank}\n')
            rank += 1
            next_5_words_16_18.append(next_word)
            doc_16_18 += [next_word]


# Test the model
test_source = pd.read_parquet('../rerank/data/group_ngram/test_source_group_ngram.parquet')
test_source = test_source[["session_id", "song_id"]]
test_sessions = test_source['session_id'].unique()
test_corpus = test_source.groupby('session_id')['song_id'].apply(list).tolist()
generate_score(test_sessions, test_corpus, args, "../rerank/test_features/ngram_score_top300.csv")


train_source = pd.read_parquet('../rerank/data/group_ngram/train_source_group_ngram.parquet')
train_source = train_source[["session_id", "song_id"]]
train_sessions = train_source['session_id'].unique()
train_corpus = train_source.groupby('session_id')['song_id'].apply(list).tolist()
generate_score(train_sessions, train_corpus, args, "../rerank/train_features/ngram_score_top300.csv")
