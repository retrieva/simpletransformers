# Download CoNLL-2003 dataset from the git repo down below.
# https://github.com/synalp/NER/tree/master/corpus/CoNLL-2003

from simpletransformers.ner import NERModel

# BERT UNCASED
model = NERModel('bert', 'bert-base-uncased', args={'overwrite_output_dir':True})
model.train_model('CoNLL-2003/eng.train')
results, model_outputs, predictions = model.eval_model('CoNLL-2003/eng.testb')
print(results)

# BERT CASED
model = NERModel('bert', 'bert-base-cased', args={'overwrite_output_dir':True})
model.train_model('CoNLL-2003/eng.train')
results, model_outputs, predictions = model.eval_model('CoNLL-2003/eng.testb')
print(results)

# ALBERT 
model = NERModel('albert', 'albert-base-v2', args={'overwrite_output_dir':True})
model.train_model('CoNLL-2003/eng.train')
results, model_outputs, predictions = model.eval_model('CoNLL-2003/eng.testb')
print(results)