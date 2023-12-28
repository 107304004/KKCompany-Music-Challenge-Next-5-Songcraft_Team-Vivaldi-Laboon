import math
import pandas as pd
import random
from collections import defaultdict
from operator import itemgetter
from tqdm import tqdm


def LoadData(train_source_path, train_target_path, test_source_path):
    train_source = pd.read_parquet(train_source_path)
    train_target = pd.read_parquet(train_target_path)
    test_source = pd.read_parquet(test_source_path)

    train_df = pd.concat([train_source, train_target, test_source], axis=0)[['session_id', 'song_id']]
    test_df = test_source[['session_id', 'song_id']]

    # remove songs in train_df which not appeared in test_df
    # candidate_songs = set(train_source['song_id'].unique().tolist()) & set(train_target['song_id'].unique().tolist()) & set(test_source['song_id'].unique().tolist())
    # train_df = train_df[train_df['song_id'].isin(candidate_songs)]
    # test_df = test_df[test_df['song_id'].isin(candidate_songs)]

    train = []
    for idx, row in tqdm(train_df.iterrows(), total=train_df.shape[0]):
        user = row['session_id']
        item = row['song_id']
        train.append([user, item])

    test = []
    for idx, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
        user = row['session_id']
        item = row['song_id']
        test.append([user, item])

    return PreProcessData(train), PreProcessData(test)


def PreProcessData(originData):
    """
    建立User-Item表，结构如下：
        {"User1": [MovieID1, MoveID2, MoveID3,...]
         "User2": [MovieID12, MoveID5, MoveID8,...]
         ...
        }
    """
    trainData = dict()
    for user, item in originData:
        trainData.setdefault(user, list())
        trainData[user].append(item)
    return trainData


class ItemCF(object):
    """ Item based Collaborative Filtering Algorithm Implementation"""

    def __init__(self, trainData, testData, similarity="cosine", norm=False):
        self._trainData = trainData
        self._testData = testData
        self._similarity = similarity
        self._isNorm = norm
        self._itemSimMatrix = dict() # 物品相似度矩阵

    def similarity(self):
        N = defaultdict(int) # 记录每个物品的喜爱人数
        for user, items in tqdm(self._trainData.items()):
            for i_idx, i in enumerate(items):
                self._itemSimMatrix.setdefault(i, dict())
                N[i] += 1
                for j_idx, j in enumerate(items):
                    # if i == j:
                    #     continue
                    self._itemSimMatrix[i].setdefault(j, 0)
                    if self._similarity == "cosine":
                        self._itemSimMatrix[i][j] += 1
                        if j_idx > i_idx:
                            self._itemSimMatrix[i][j] += 5 * (1 - ((j_idx - i_idx) / len(items)))
                    elif self._similarity == "dot":
                        self._itemSimMatrix[i][j] += 1
                        if j_idx > i_idx:
                            self._itemSimMatrix[i][j] += 5 * (1 - ((j_idx - i_idx) / len(items)))
                    elif self._similarity == "iuf":
                        self._itemSimMatrix[i][j] += 1. / math.log1p(len(items) * 1.)
        if self._similarity == "dot":
            for i, related_items in self._itemSimMatrix.items():
                for j, cij in related_items.items():
                    self._itemSimMatrix[i][j] = cij
        else:
            for i, related_items in self._itemSimMatrix.items():
                for j, cij in related_items.items():
                    self._itemSimMatrix[i][j] = cij / math.sqrt(N[i]*N[j])
        # 是否要标准化物品相似度矩阵
        if self._isNorm:
            for i, relations in self._itemSimMatrix.items():
                # print(i, relations)
                max_num = relations[max(relations, key=relations.get)]
                # 对字典进行归一化操作之后返回新的字典
                self._itemSimMatrix[i] = {k : v/max_num for k, v in relations.items()}

    def recommend(self, user, L=20, N=5, K=400):
        """
        :param user: 被推荐的用户user
        :param L: 只看用戶最後L個商品
        :param N: 推荐的商品个数
        :param K: 查找的最相似的用户个数
        :return: 按照user对推荐物品的感兴趣程度排序的N个商品
        """
        recommends = dict()
        # 先获取user的喜爱物品列表
        items = self._testData[user]
        for item in items[-L:]:
            # 对每个用户喜爱物品在物品相似矩阵中找到与其最相似的K个
            for i, sim in sorted(self._itemSimMatrix[item].items(), key=itemgetter(1), reverse=True)[:K]:
                if len(set(items))>17:
                    if i in items:
                        continue  # 如果与user喜爱的物品重复了，则直接跳过
                    recommends.setdefault(i, 0.)
                    recommends[i] += sim
                else:
                    recommends.setdefault(i, 0.)
                    recommends[i] += sim
        # 根据被推荐物品的相似度逆序排列，然后推荐前N个物品给到用户
        # List: [item1, item2, item3, ...]
        topN = list(dict(sorted(recommends.items(), key=itemgetter(1), reverse=True)[:N]).keys())

        # if no item recommended, random sample interacted item from a user
        while len(topN)<N:
            # print(f"no result at {user}: {topN}")
            topN.append(random.sample(items, 1))

        return topN 

    def train(self):
        self.similarity()


if __name__ == "__main__":
    print("Loading data...")
    train, test = LoadData("data/label_train_source.parquet",
                           "data/label_train_target.parquet",
                           "data/label_test_source.parquet")
    # train, test = LoadData("small_data/label_train_source.parquet", "small_data/label_train_target.parquet", "small_data/label_test_source.parquet")
    print("train data size: %d, test data size: %d" % (len(train), len(test)))
    print('Start training...')
    ItemCF = ItemCF(train, test, similarity='cosine')
    ItemCF.train()

    print("Start testing...")
    sub = []
    for test_user in tqdm(ItemCF._testData.keys()):
        res = ItemCF.recommend(test_user, L=5, N=5)
        res.insert(0, test_user)
        sub.append(res)
    # print(sub)

    sub_df = pd.DataFrame(columns=['session_id', 'top1', 'top2', 'top3', 'top4', 'top5'],
                          data = sub)
    sub_df.to_csv("subs/itemcf_onedir_timebased_5.csv", index=False)
