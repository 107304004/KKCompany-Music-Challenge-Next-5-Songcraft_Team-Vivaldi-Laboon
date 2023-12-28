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

# Sample training corpus
# corpus = [
#     ["here", "is", "the", "text", "for", "document1"],
#     ["this", "is", "the", "text", "for", "document2"],
#     ["we", "have", "some", "texts", "for", "document3"]
# ]
# Read corpus
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


print("generate song_id")
# Test the model
# context = ["here", "is"]  # Provide a context for next-word prediction
test_source = pd.read_parquet("../data/ngram_test_source.parquet").sort_values(["session_id", "unix_played_at"])
test_sessions = test_source['session_id'].unique()
test_corpus = test_source.groupby('session_id')['song_id'].apply(list).tolist()
sub = []
num_of_random = 0
for s_id, doc in tzip(test_sessions, test_corpus):

    # next 5 words by song 18~20
    doc_18_20 = doc[-(args.n - 1):]
    next_5_words_18_20 = []
    while len(next_5_words_18_20) < 5:
        # 4gram
        next_word = model.generate(1, doc_18_20[-(args.n - 1):])
        # 3gram
        if next_word == '</s>':
            next_word = model.generate(1, doc_18_20[-(args.n - 2):])
        next_5_words_18_20.append(next_word)
        doc_18_20 += [next_word]

    # next 5 words by song 17~19
    doc_17_19 = doc[-(args.n - 0):-1]
    next_5_words_17_19 = []
    while len(next_5_words_17_19) < 5:
        # 4gram
        next_word = model.generate(1, doc_17_19[-(args.n - 1):])
        # if next_word == doc[-1]:
        #     break
        # 3gram
        if next_word == '</s>':
            next_word = model.generate(1, doc_17_19[-(args.n - 2):])
        next_5_words_17_19.append(next_word)
        doc_17_19 += [next_word]

    # next 5 words by song 16~18
    doc_16_18 = doc[-(args.n + 1):-2]
    next_5_words_16_18 = []
    while len(next_5_words_16_18) < 5:
        # 4gram
        next_word = model.generate(1, doc_16_18[-(args.n - 1):])
        # if next_word == doc[-2]:
        #     break
        # 3gram
        if next_word == '</s>':
            next_word = model.generate(1, doc_16_18[-(args.n - 2):])
        next_5_words_16_18.append(next_word)
        doc_16_18 += [next_word]

    # # next 5 words by song 15~17
    # doc_15_17 = doc[-(args.n + 2):-3]
    # next_5_words_15_17 = []
    # while len(next_5_words_15_17) < 5:
    #     # 4gram
    #     next_word = model.generate(1, doc_15_17[-(args.n - 1):])
    #     # if next_word == doc[-2]:
    #     #     break
    #     # 3gram
    #     if next_word == '</s>':
    #         next_word = model.generate(1, doc_15_17[-(args.n - 2):])
    #     next_5_words_15_17.append(next_word)
    #     doc_15_17 += [next_word]

    next_words = next_5_words_18_20 + next_5_words_17_19 + next_5_words_16_18 
    top5 = [x for x in next_words if x != '</s>'][:5]

    while len(top5) < 5:
        top5.append(random.choice(doc[:20]))
        num_of_random += 1
        # top5.append('</s>')
    top5.insert(0, s_id)
    sub.append(top5)

print(f'numbers of random submit: {num_of_random}')
sub_df = pd.DataFrame(columns=['session_id', 'top1', 'top2', 'top3', 'top4', 'top5'],
                      data=sub)
sub_df.to_csv(args.output_path, index=False)
