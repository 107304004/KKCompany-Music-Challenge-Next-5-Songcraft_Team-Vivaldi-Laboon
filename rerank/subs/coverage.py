import pandas as pd
import random
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--sub_path", "-s", type=str)
args = parser.parse_args()

test_df = pd.read_parquet("../../data/label_test_source.parquet")
sub = pd.read_csv(args.sub_path)

song_meta = pd.read_parquet("../../data/meta_song.parquet")['song_id'].tolist()
total_unique_values = sub.drop(['session_id'], axis=1).to_numpy().flatten().tolist()
song_for_cover = list(set(song_meta) - set(total_unique_values))

# Function to replace duplicates with new items
# i = 0
def replace_duplicates(row):
    seen_items = set()
    new_top = []
    for top in row[1:]:
        if top not in seen_items:
            seen_items.add(top)
            new_top.append(top)
        else:
            new_top.append(random.choice(song_for_cover))
            # i += 1
    return new_top

# Apply the function to each row
sub['new_tops'] = sub.apply(replace_duplicates, axis=1)
# Split the 'new_tops' column back into individual columns
sub[['new_top1', 'new_top2', 'new_top3', 'new_top4', 'new_top5']] = pd.DataFrame(sub['new_tops'].tolist(), index=sub.index)
# Drop the original 'top1' through 'top5' columns and the temporary 'new_tops' column
sub.drop(['top1', 'top2', 'top3', 'top4', 'top5', 'new_tops'], axis=1, inplace=True)
sub = sub.rename(columns={'new_top1':'top1', 'new_top2':'top2', 'new_top3':'top3', 'new_top4':'top4', 'new_top5':'top5'})
print(sub.head())
print('num of unique songs:')
print(len(set(sub.drop(['session_id'], axis=1).to_numpy().flatten().tolist())))

sub.to_csv(f'cov_{args.sub_path}', index=False)
