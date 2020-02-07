import pandas as pd
import numpy as np
import sklearn
import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir) + '/../')
from simpletransformers.classification.classification_model import ClassificationModel as ClassificationModel


def main():
    """
    data is necessary to be pandas data frame and to meets either of following
    1. df has 'text' column and 'labels' column and contains text data in 'text' column and label data in 'label' column
    2. the first column of df is text data and second column is label data.
    this example is second type.
    """

    df_train = pd.read_csv('/data/bert/livedoor_news/train.tsv', delimiter='\t')
    df_dev = pd.read_csv('/data/bert/livedoor_news/dev.tsv', delimiter='\t')

    # only number label is allowed so convert.
    labels, uniques = pd.factorize(df_train.loc[:, 'label'])
    df_train.loc[:, 'label'] = labels.astype('int')
    dev_labels = uniques.get_indexer(df_dev.loc[:, 'label']).astype('int')
    df_dev.loc[:, 'label'] = dev_labels
    label_num = len(uniques)

    # model_path for fine-tune
    model_path = '/data/bert/model/wiki_asahi/512/pytorch_model-300000.bin'

    # model_path for evaluate
    # model_path = './outputs/'
    config_path = '/data/bert/model/512/bert_config.json'
    tokenizer_path = '/data/bert/model/wiki_asahi/wiki_asahi.model'

    model = ClassificationModel('bert', model_path, config_path, tokenizer_path, num_labels=label_num,
                                args={'output_dir': './livedoor', 'num_train_epochs': 5, 'max_seq_length': 512, 'train_batch_size': 16,
                                      'overwrite_output_dir': True})

    model.train_model(df_train, show_running_loss=False)

    # evaluate by model.eval_model()
    result, model_outputs, wrong_prediction = model.eval_model(df_dev, acc=sklearn.metrics.accuracy_score)
    print(result)

    # evaluate by pred
    pred, _ = model.predict(df_dev.loc[:, 'text'])
    print(sklearn.metrics.classification_report(pred, dev_labels, target_names=uniques))


if __name__ == "__main__":
    main()
