import pandas as pd
import numpy as np
import xgboost as xgb
from tqdm import tqdm


# label
target = pd.read_parquet('../data/label_train_target.parquet').sort_values(by=['session_id', 'listening_order'])[['session_id', 'song_id']]
target['session_id_song_id'] = target['session_id'].apply(str) + "_" + target['song_id']
target['label'] = 1
target = target[['session_id_song_id', 'label']]


# training features
print("Preparing training features...")
ngram = pd.read_csv('train_features/ngram_score_top100.csv')
ngram['ngram_score'] = ngram['ngram_rank'].apply(lambda x: 1/x)
ngram = ngram.drop_duplicates(subset=['session_id_song_id'])
itemcf = pd.read_csv('train_features/itemcf_score_top25.csv')
bm25 = pd.read_csv('train_features/bm25_score_top25.csv')
gru = pd.read_csv('train_features/gru_score_top25.csv')

df_train = ngram.merge(itemcf, how='outer', on='session_id_song_id').merge(bm25, how='outer', on='session_id_song_id').merge(gru, how='outer', on='session_id_song_id')
df_train = df_train.merge(target, how='left', on='session_id_song_id')
df_train['label'] = df_train['label'].fillna(0)
df_train['session_id'] = df_train['session_id_song_id'].apply(lambda x: int(x.split('_')[0]))
df_train['song_id'] = df_train['session_id_song_id'].apply(lambda x: x.split('_')[1])

print(df_train.shape)
print(df_train.head())
# print(df_train.isnull().sum())
print(df_train['label'].value_counts())
# import IPython;IPython.embed(colors='linux');exit(1)

df_train = df_train.fillna(0)
df_train = df_train.sort_values(by=['session_id'])
qid_train = df_train['session_id'].to_numpy()
X_train = df_train[['ngram_score', 'itemcf_score', 'bm25_score', 'gru_score']].to_numpy()
y_train = df_train['label'].to_numpy()


# testing features
print("Preparing testing features...")
ngram_test = pd.read_csv('test_features/ngram_score_top100.csv')
ngram_test['ngram_score'] = ngram_test['ngram_rank'].apply(lambda x: 1/x)
itemcf_test = pd.read_csv('test_features/itemcf_score_top100.csv')
bm25_test = pd.read_csv('test_features/bm25_score_top100.csv')
gru_test = pd.read_csv('test_features/gru_score_top100.csv')

df_test = ngram_test.merge(itemcf_test, how='outer', on='session_id_song_id').merge(bm25_test, how='outer', on='session_id_song_id').merge(gru_test, how='outer', on='session_id_song_id')
df_test['session_id'] = df_test['session_id_song_id'].apply(lambda x: int(x.split('_')[0]))
df_test['song_id'] = df_test['session_id_song_id'].apply(lambda x: x.split('_')[1])

df_test = df_test.fillna(0)
df_test = df_test.sort_values(by=['session_id'])
qid_test = df_test['session_id'].to_numpy()
X_test = df_test[['ngram_score', 'itemcf_score', 'bm25_score', 'gru_score']].to_numpy()


# create and train the classifier
print('Training xgbranker...')
ranker = xgb.XGBRanker(random_state=0,
                       tree_method="hist", 
                       lambdarank_num_pair_per_sample=10, 
                       objective="rank:ndcg", 
                       lambdarank_pair_method="topk")
ranker.fit(X_train, y_train, qid=qid_train)


# predictions
y_pred = ranker.predict(X_test)
df_test['score'] = y_pred
df_test.to_csv("ensemble_t25t100_result.csv")
df_test = df_test.groupby(['session_id']).apply(
        lambda x: x.sort_values(by=['score', 'ngram_score'], ascending = False)
).reset_index(drop=True)
print(df_test.head(15))


# output
print("Saving result...")
test_dict = dict()
for idx, row in tqdm(df_test.iterrows(), total=df_test.shape[0]):
    test_dict.setdefault(row['session_id'], list())
    test_dict[row['session_id']].append(row['song_id'])
sub = []
for s_id, top5 in tqdm(test_dict.items()):
    top5 = top5[:5]
    top5.insert(0, s_id)
    sub.append(top5)

sub_df = pd.DataFrame(columns=['session_id', 'top1', 'top2', 'top3', 'top4', 'top5'], data = sub)
sub_df.to_csv('./subs/group_ngram/xgboost_t25t100.csv', index=False)
