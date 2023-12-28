from recbole.quick_start import load_data_and_model
from recbole.trainer import Trainer
import pandas as pd
import argparse

from util.predictor import sequential_predict, sequential_predict_score, sequential_predict_ar, general_predict


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="./saved/GRU4Rec-Dec-07-2023_01-15-58.pth", help="name of saved models")
    parser.add_argument("--train", type=int, default=0, help="if train the model")
    parser.add_argument("--test_path", "-t", type=str, default="../data/label_test_source.parquet", help="test_file")
    parser.add_argument("--output_path", "-o", type=str, default="../subs/gru4rec.csv", help="output path")
    args = parser.parse_args()

    config, model, dataset, train_data, valid_data, test_data = load_data_and_model(model_file=args.model)

    if args.train:
        trainer = Trainer(config, model)
        best_valid_score, best_valid_result = trainer.fit(
            train_data,
            valid_data,
            saved=True,
            show_progress=config["show_progress"]
        )

    # testing
    test_df = pd.read_parquet(args.test_path)
    # sub = sequential_predict(test_df, dataset, model)
    # sub = sequential_predict_ar(test_df, dataset, model)
    sub = sequential_predict_score(test_df, dataset, model)
    # sub = general_predict(test_df, dataset, model)

    # output
    sub_df = pd.DataFrame(columns=['session_id_song_id', 'gru_score', 'gru_rank'], data = sub)
    sub_df.to_csv(args.output_path, index=False)
