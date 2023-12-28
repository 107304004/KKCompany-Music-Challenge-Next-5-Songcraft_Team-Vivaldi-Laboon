from tqdm import tqdm
from tqdm.contrib import tzip
import torch
import numpy as np
from recbole.data.interaction import Interaction


def sequential_predict(test_df, dataset, model):
    test_dict = dict()
    for idx, row in test_df.iterrows():
        test_dict.setdefault(row['session_id'], list())
        if row['song_id'] in dataset.field2token_id[dataset.iid_field].keys():
            test_dict[row['session_id']].append(row['song_id'])

    unique_test_sessions = test_df['session_id'].unique()
    sub = []
    for s in tqdm(unique_test_sessions):
        if len(test_dict[s]) < 1:
            continue
        
        if len(set(test_dict[s])) > 17:
            # transform song_id in test item seq
            item_seq = torch.tensor(dataset.token2id(dataset.iid_field, test_dict[s]))
            # interaction data format
            inter = Interaction({
                "song_id_list": item_seq.reshape(1, -1),
                "item_length": torch.tensor([len(item_seq)])
            })
            # predict
            score = model.full_sort_predict(inter.to(model.device))
            score[:, 0] = -np.inf  # padded item score = -inf
            # top5
            topk_score, topk_iid_list = torch.topk(score, 25)
            # transform back to song_id
            predicted_item_list = dataset.id2token(
                dataset.iid_field, topk_iid_list.tolist()
            ).tolist()[0]
            # not recommend old items
            predicted_item_list = np.delete(predicted_item_list, np.isin(predicted_item_list, test_dict[s]))[:5].tolist()
            # save result
            predicted_item_list.insert(0, s)
            sub.append(predicted_item_list)
        else:
            # transform song_id in test item seq
            item_seq = torch.tensor(dataset.token2id(dataset.iid_field, test_dict[s]))
            # interaction data format
            inter = Interaction({
                "song_id_list": item_seq.reshape(1, -1),
                "item_length": torch.tensor([len(item_seq)])
            })
            # predict
            score = model.full_sort_predict(inter.to(model.device))
            score[:, 0] = -np.inf  # padded item score = -inf
            # top5
            topk_score, topk_iid_list = torch.topk(score, 5)
            # transform back to song_id
            predicted_item_list = dataset.id2token(
                dataset.iid_field, topk_iid_list.tolist()
            ).tolist()[0]
            # save result
            predicted_item_list.insert(0, s)
            sub.append(predicted_item_list)

    return sub


def sequential_predict_ar(test_df, dataset, model):
    test_dict = dict()
    for idx, row in test_df.iterrows():
        test_dict.setdefault(row['session_id'], list())
        test_dict[row['session_id']].append(row['song_id'])

    unique_test_sessions = test_df['session_id'].unique()
    sub = []
    for s in tqdm(unique_test_sessions):
        if len(test_dict[s]) < 1:
            continue
        top5 = []
        for i in range(5):
            # transform song_id in test item seq
            item_seq = torch.tensor(dataset.token2id(dataset.iid_field, test_dict[s][-20:]))
            # print(item_seq)
            # interaction data format
            inter = Interaction({
                "song_id_list": item_seq.reshape(1, -1),
                "item_length": torch.tensor([5])
            })
            # predict
            score = model.full_sort_predict(inter.to(model.device))
            score[:, 0] = -np.inf  # padded item score = -inf
            # top5
            topk_score, topk_iid_list = torch.topk(score, 1)
            # transform back to song_id
            predicted_item = dataset.id2token(
                dataset.iid_field, topk_iid_list.tolist()
            ).tolist()[0][0]
            # save result
            top5.append(predicted_item)
            # append new item to predict next
            test_dict[s].append(predicted_item)

        top5.insert(0, s)
        # print(top5)
        sub.append(top5)

    return sub


def general_predict(test_df, dataset, model):
    test_dict = dict()
    for idx, row in test_df.iterrows():
        test_dict.setdefault(row['session_id'], list())
        test_dict[row['session_id']].append(row['song_id'])

    unique_test_sessions = test_df['session_id'].unique()
    sub = []
    for s in tqdm(unique_test_sessions):
        if len(set(test_dict[s])) > 17:
            uid = torch.tensor(dataset.token2id(dataset.uid_field, [str(s)]))
            # interaction data format
            inter = Interaction({
                "session_id": uid,
            })
            # predict
            score = model.full_sort_predict(inter.to(model.device))
            score[0] = -np.inf  # padded item score = -inf
            # top5
            topk_score, topk_iid_list = torch.topk(score, 5)
            # transform back to song_id
            predicted_item_list = dataset.id2token(
                dataset.iid_field, topk_iid_list.tolist()
            ).tolist()
            predicted_item_list = np.delete(predicted_item_list, np.isin(predicted_item_list, test_dict[s]))[:5].tolist()
            # save result
            predicted_item_list.insert(0, s)
            sub.append(predicted_item_list)
        else:
            uid = torch.tensor(dataset.token2id(dataset.uid_field, [str(s)]))
            # interaction data format
            inter = Interaction({
                "session_id": uid,
            })
            # predict
            score = model.full_sort_predict(inter.to(model.device))
            score[0] = -np.inf  # padded item score = -inf
            # top5
            topk_score, topk_iid_list = torch.topk(score, 5)
            # transform back to song_id
            predicted_item_list = dataset.id2token(
                dataset.iid_field, topk_iid_list.tolist()
            ).tolist()
            # save result
            predicted_item_list.insert(0, s)
            sub.append(predicted_item_list)

    return sub


def sequential_predict_score(test_df, dataset, model):
    test_dict = dict()
    for idx, row in test_df.iterrows():
        test_dict.setdefault(row['session_id'], list())
        if row['song_id'] in dataset.field2token_id[dataset.iid_field].keys():
            test_dict[row['session_id']].append(row['song_id'])

    unique_test_sessions = test_df['session_id'].unique()
    sub = []
    for s in tqdm(unique_test_sessions):
        if len(test_dict[s]) < 1:
            continue
        
        # transform song_id in test item seq
        item_seq = torch.tensor(dataset.token2id(dataset.iid_field, test_dict[s]))
        # interaction data format
        inter = Interaction({
            "song_id_list": item_seq.reshape(1, -1),
            "item_length": torch.tensor([len(item_seq)])
        })
        # predict
        score = model.full_sort_predict(inter.to(model.device))
        score[:, 0] = -np.inf  # padded item score = -inf
        # top5
        topk_score, topk_iid_list = torch.topk(score, 300)
        # transform back to song_id
        predicted_item_list = dataset.id2token(
            dataset.iid_field, topk_iid_list.tolist()
        ).tolist()[0]

        # import IPython;IPython.embed(colors='linux');exit(1)
        rank = 1
        for song_id, score in zip(predicted_item_list, topk_score[0]):
            if len(set(test_dict[s])) > 17:
                if song_id in test_dict[s]:
                    continue
                sub.append([str(s)+"_"+song_id, score.item(), rank])
                rank += 1
            else:
                sub.append([str(s)+"_"+song_id, score.item(), rank])
                rank += 1

    return sub
