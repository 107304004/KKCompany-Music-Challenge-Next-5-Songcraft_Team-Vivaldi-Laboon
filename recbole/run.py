import argparse
import logging
from logging import getLogger
from recbole.config import Config
from recbole.data import create_dataset, data_preparation
from recbole.utils import init_seed, init_logger
from recbole.model.sequential_recommender import GRU4Rec, SASRec, BERT4Rec
from recbole.model.sequential_recommender import GRU4RecF, SASRecF, S3Rec, SASRecD
from recbole.model.general_recommender import LightGCN, SimpleX, SLIMElastic
from recbole.trainer import Trainer
from recbole.data.interaction import Interaction

import pandas as pd
import numpy as np
import torch

from tqdm import tqdm
from util.predictor import sequential_predict, general_predict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="GRU4Rec", help="name of models")
    parser.add_argument("--dataset", "-d", type=str, default="kkdatagame", help="name of datasets")
    parser.add_argument("--config_files", "-c", type=str, default="configs/kkdatagame.yaml", help="config file")
    parser.add_argument("--test_path", "-t", type=str, default="../data/label_test_source.parquet", help="test_file")
    parser.add_argument("--output_path", "-o", type=str, default="../subs/gru4rec.csv", help="output path")

    args, _ = parser.parse_known_args()

    config_file_list = (args.config_files.strip().split(" ") if args.config_files else None)

    parameter_dict = {
    #     'neg_sampling': None,
    #     'train_neg_sample_args': None,
    }

    config = Config(
        model=args.model,
        dataset=args.dataset,
        config_file_list=config_file_list,
        config_dict=parameter_dict,
    )

    init_seed(config['seed'], config['reproducibility'])
    init_logger(config)
    logger = getLogger()
    logger.info(config)

    dataset = create_dataset(config)
    logger.info(dataset)
    train_data, valid_data, test_data = data_preparation(config, dataset)

    if args.model == 'GRU4Rec':
        model = GRU4Rec(config, train_data.dataset).to(config['device'])
    if args.model == 'GRU4RecF':
        model = GRU4RecF(config, train_data.dataset).to(config['device'])
    if args.model == 'SASRec':
        model = SASRec(config, train_data.dataset).to(config['device'])
    if args.model == 'SASRecF':
        model = SASRecF(config, train_data.dataset).to(config['device'])
    if args.model == 'SASRecD':
        model = SASRecD(config, train_data.dataset).to(config['device'])
    if args.model == 'S3Rec':
        model = S3Rec(config, train_data.dataset).to(config['device'])
    if args.model == 'BERT4Rec':
        model = BERT4Rec(config, train_data.dataset).to(config['device'])
    if args.model == 'LightGCN':
        model = LightGCN(config, train_data.dataset).to(config['device'])
    if args.model == 'SimpleX':
        model = SimpleX(config, train_data.dataset).to(config['device'])
    if args.model == 'SLIMElastic':
        model = SLIMElastic(config, train_data.dataset).to(config['device'])
    logger.info(model)

    # training
    trainer = Trainer(config, model)
    best_valid_score, best_valid_result = trainer.fit(
        train_data,
        valid_data,
        saved=True,
        show_progress=config["show_progress"]
    )

    # testing
    test_df = pd.read_parquet(args.test_path)
    # test_df = pd.read_parquet("../small_data/label_test_source.parquet")
    sub = sequential_predict(test_df, dataset, model)
    # sub = general_predict(test_df, dataset, model)
    # print(sub)

    # output
    sub_df = pd.DataFrame(columns=['session_id', 'top1', 'top2', 'top3', 'top4', 'top5'],
                          data = sub)
    sub_df.to_csv(args.output_path, index=False)
