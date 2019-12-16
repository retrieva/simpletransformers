import pandas as pd
import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir) + '/../')
from simpletransformers.classification.classification_model import ClassificationModel as ClassificationModel



# Train and Evaluation data needs to be in a Pandas Dataframe of two columns. The first column is the text with type str, and the second column is the label with type int.
train_data = [['Example sentence belonging to class 1', 1], ['Example sentence belonging to class 0', 0]]
train_df = pd.DataFrame(train_data)

eval_data = [['Example eval sentence belonging to class 1', 1], ['Example eval sentence belonging to class 0', 0]]
eval_df = pd.DataFrame(eval_data)

# Create a ClassificationModel
model = ClassificationModel('albert', 'albert-base-v2') # You can set class weights by using the optional weight argument

# Train the model
model.train_model(train_df)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)
pred = model.predict(eval_df.iloc[:, 0])
print(pred)