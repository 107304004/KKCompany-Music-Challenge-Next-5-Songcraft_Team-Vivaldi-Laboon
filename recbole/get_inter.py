import pandas as pd


# kkdatagame.inter
train_df = pd.read_parquet("../data/label_train_source.parquet")
train_target = pd.read_parquet("../data/label_train_target.parquet")
test_df = pd.read_parquet("../data/label_test_source.parquet")
# train_df = pd.read_parquet("../small_data/label_train_source.parquet")
# train_target = pd.read_parquet("../small_data/label_train_target.parquet")
# test_df = pd.read_parquet("../small_data/label_test_source.parquet")
# train_df = pd.read_parquet("../data_preprocess/sep_month_data/oct_train_source.parquet")
# train_target = pd.read_parquet("../data_preprocess/sep_month_data/oct_train_target.parquet")
# test_df = pd.read_parquet("../data_preprocess/sep_month_data/oct_test_source.parquet")

inter_df = pd.concat([train_df, train_target, test_df], axis=0)
# inter_df = test_df
inter_df = inter_df.drop(['listening_order'], axis=1)
try:
    inter_df = inter_df.drop(['played_at_month'], axis=1)
except:
    pass

inter_df = inter_df.sort_values(['session_id', 'unix_played_at'])
inter_df = inter_df.rename(columns={
    'session_id': 'session_id:token',
    'song_id': 'song_id:token',
    'unix_played_at': 'timestamp:float',
    'play_status': 'play_status:token',
    'login_type': 'login_type:token'
})

print(inter_df.head())
inter_df.to_csv('dataset/kkdatagame/kkdatagame.inter', sep='\t', index=False)
# inter_df.to_csv('dataset/small/small.inter', sep='\t', index=False)
# inter_df.to_csv('dataset/test/test.inter', sep='\t', index=False)
# inter_df.to_csv('dataset/oct/oct.inter', sep='\t', index=False)


'''
# kkdatagame.item
song = pd.read_parquet('../data/meta_song.parquet')
song = song.replace('nan', 'None')
song_composer = pd.read_parquet('../data/meta_song_composer.parquet')
composer = dict()
for idx, row in song_composer.iterrows():
    composer.setdefault(row['song_id'], "")
    composer[row['song_id']] += (row['composer_id'] + " ")
song_genre = pd.read_parquet('../data/meta_song_genre.parquet')
genre = dict()
for idx, row in song_genre.iterrows():
    genre.setdefault(row['song_id'], "")
    genre[row['song_id']] += (row['genre_id'] + " ")
song_lyricist = pd.read_parquet('../data/meta_song_lyricist.parquet')
lyricist = dict()
for idx, row in song_lyricist.iterrows():
    lyricist.setdefault(row['song_id'], "")
    lyricist[row['song_id']] += (row['lyricist_id'] + " ")
song_producer = pd.read_parquet('../data/meta_song_producer.parquet')
producer = dict()
for idx, row in song_producer.iterrows():
    producer.setdefault(row['song_id'], "")
    producer[row['song_id']] += (row['producer_id'] + " ")
song_titletext = pd.read_parquet('../data/meta_song_titletext.parquet')
title_text = dict()
for idx, row in song_titletext.iterrows():
    title_text.setdefault(row['song_id'], "")
    title_text[row['song_id']] += (row['title_text_id'] + " ")

song['composer_id'] = song['song_id'].map(composer)
song['genre_id'] = song['song_id'].map(genre)
song['lyricist_id'] = song['song_id'].map(lyricist)
song['producer_id'] = song['song_id'].map(producer)
song['title_text_id'] = song['song_id'].map(title_text)
song = song.fillna('None')

song = song.rename(columns={
    'song_id': 'song_id:token',
    'artist_id': 'artist_id:token',
    'song_length': 'song_length:float',
    'album_id': 'album_id:token',
    'language_id': 'language_id:token',
    'album_month': 'album_month:token',
    'composer_id': 'composer_id:token_seq',
    'genre_id': 'genre_id:token_seq',
    'lyricist_id': 'lyricist_id:token_seq',
    'producer_id': 'producer_id:token_seq',
    'title_text_id': 'title_text_id:token_seq'
})

print(song.head())
# song.to_csv('dataset/kkdatagame/kkdatagame.item', sep='\t', index=False)
song.to_csv('dataset/small/small.item', sep='\t', index=False)
'''
