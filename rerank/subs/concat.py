import pandas as pd
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--s1")
parser.add_argument("--s2")
parser.add_argument("--o")
args = parser.parse_args()

test_df = pd.read_parquet("../../data/label_test_source.parquet")
s1 = pd.read_csv(args.s1)
s2 = pd.read_csv(args.s2)
sub = pd.concat([s1, s2], axis=0)
print(sub.shape)

# null value
if sum(sub.isnull().sum()):
    print(f'na: {sub.isnull().sum()}')
sub.to_csv(args.o, index=False)

sub = sub.melt('session_id', var_name='rank', value_name='song_id')
# nunique in sub
print(sub.groupby('session_id')['song_id'].nunique().value_counts())
# Merge dataframes
merged_df = pd.merge(sub, test_df[['session_id', 'song_id']].drop_duplicates(), on=['session_id', 'song_id'], how='left', indicator=True)
merged_df['appeared'] = (merged_df['_merge'] == 'both').astype(int)
merged_df = merged_df[['session_id', 'song_id', 'appeared']]
print('submission answer in test source:')
tea = test_df[test_df['song_id'].isin(sub['song_id'].unique())]
teas = test_df[test_df['session_id'].isin(tea['session_id'])]['song_id'].unique()
print(merged_df.groupby('session_id')['appeared'].sum().value_counts())

print('test source songs nunique:')
print(test_df['song_id'].nunique())
print('submission ans songs nunique:')
print(sub['song_id'].nunique())
print('test+sub songs nunique:')
print(pd.concat([test_df['song_id'], sub['song_id']], axis=0).nunique())
