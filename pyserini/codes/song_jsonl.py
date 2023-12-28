import pandas as pd
import json
from tqdm import tqdm

train_df = pd.read_parquet('../data/label_train_source.parquet')
train_ta = pd.read_parquet('../data/label_train_target.parquet')
test_df = pd.read_parquet('../data/label_test_source.parquet')
df = pd.concat([train_df, train_ta, test_df], axis=0)[['session_id', 'song_id']]
song = pd.read_parquet('../data/meta_song.parquet')
song = song.applymap(str)
song_composer = pd.read_parquet('../data/meta_song_composer.parquet')
song_genre = pd.read_parquet('../data/meta_song_genre.parquet')
song_lyricist = pd.read_parquet('../data/meta_song_lyricist.parquet')
song_producer = pd.read_parquet('../data/meta_song_producer.parquet')
song_titletext = pd.read_parquet('../data/meta_song_titletext.parquet')


data = dict()
for idx, row in tqdm(song.iterrows(), total=song.shape[0]):
    data[row['song_id']] = ""
    data[row['song_id']] = data[row['song_id']] + row['artist_id'].replace(".", "_") + " "
    data[row['song_id']] = data[row['song_id']] + row['artist_id'].replace(".", "_") + "_"  # artist_album
    # data[row['song_id']] = data[row['song_id']] + row['song_length'] + " "
    data[row['song_id']] = data[row['song_id']] + row['album_id'].replace(".", "_") + " "
    data[row['song_id']] = data[row['song_id']] + row['language_id'].replace(".", "_") + " "
    data[row['song_id']] = data[row['song_id']] + row['album_month'][:3] + " "

for idx, row in tqdm(df.iterrows(), total=df.shape[0]):
    data[row['song_id']] = data[row['song_id']] + 'session' + str(row['session_id']) + " "

for idx, row in tqdm(song_composer.iterrows(), total=song_composer.shape[0]):
    try:
        data[row['song_id']] = data[row['song_id']] + row['composer_id'] + " "
    except:
        pass
for idx, row in tqdm(song_genre.iterrows(), total=song_genre.shape[0]):
    try:
        data[row['song_id']] = data[row['song_id']] + row['genre_id'] + " "
    except:
        pass
for idx, row in tqdm(song_lyricist.iterrows(), total=song_lyricist.shape[0]):
    try:
        data[row['song_id']] = data[row['song_id']] + row['lyricist_id'] + " "
    except:
        pass
for idx, row in tqdm(song_producer.iterrows(), total=song_producer.shape[0]):
    try:
        data[row['song_id']] = data[row['song_id']] + row['producer_id'] + " "
    except:
        pass
for idx, row in tqdm(song_titletext.iterrows(), total=song_titletext.shape[0]):
    try:
        data[row['song_id']] = data[row['song_id']] + row['title_text_id'] + " "
    except:
        pass


# output_file = 'data/collection/collection.jsonl'
# output_file = 'data/test_collection/test_collection.jsonl'
output_file = 'data/session_song_collection/session_song_collection.jsonl'
with open(output_file, 'w') as f:
    for song_id, content in tqdm(data.items()):
        content = content.replace("nan", "")
        content = content.replace("None", "")
        content = content.replace("Non", "")
        json_data = {"id": song_id, "contents": content}
        f.write(json.dumps(json_data) + "\n")
