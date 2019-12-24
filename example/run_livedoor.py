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

    df_train = pd.read_csv('/data/language/corpus/livedoor_news/train.tsv', delimiter='\t')
    df_dev = pd.read_csv('/data/language/corpus/livedoor_news/dev.tsv', delimiter='\t')

    # only number label is allowed so convert.
    labels, uniques = pd.factorize(df_train.loc[:, 'label'])
    df_train.loc[:, 'label'] = labels.astype('int')
    dev_labels = uniques.get_indexer(df_dev.loc[:, 'label']).astype('int')
    df_dev.loc[:, 'label'] = dev_labels
    label_num = len(uniques)

    # model_path for fine-tune
    model_path = '/data/language/bert/model_wiki_128/model.pytorch-1400000'

    # model_path for evaluate
    # model_path = './outputs/'
    config_path = '/data/language/bert/model_wiki_128/bert_config.json'
    tokenizer_path = '/data/language/bert/model_wiki_128/wiki-ja.model'

    model = ClassificationModel('bert', model_path, config_path, tokenizer_path, num_labels=label_num)

    model.train_model(df_train, show_running_loss=False)

    # evaluate by model.eval_model()
    result, model_outputs, wrong_prediction = model.eval_model(df_dev, acc=sklearn.metrics.accuracy_score)
    print(result)

    # evaluate by pred
    pred, _ = model.predict(df_dev.loc[:, 'text'])
    print(sklearn.metrics.classification_report(pred, dev_labels, target_names=uniques))


if __name__ == "__main__":
    main()
