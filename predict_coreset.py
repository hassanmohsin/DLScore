# Script for predicting coreset using top 100 dlscore models based on their validation performance
# Md Mahmudulla Hassan
# Last modified: 07/01/2019

import json
import os
from keras.models import model_from_json
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import pandas as pd

# Configure tensorflow memory management
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

# Read the result file
results = dict()
with open('dicts.json', 'r') as f:
    results = json.load(f)

# Get the performances on the validation set and sort
valid = [[k, res['pearson_valid']] for k, res in results.items()]
valid.sort(key=lambda x: x[1], reverse=True)

# Get the model names (basically the layers)
model_names = [m[0] for m in valid[:100]]

# Load the test data
test_x = np.load(os.path.join('data', 'test_x.npy'))
test_y = np.load(os.path.join('data', 'test_y.npy'))
results = {'true_values': test_y.tolist()}

# Get the predictions using the top models
for model_name in tqdm(model_names):
    with open(os.path.join('train_dir', model_name, 'model.json'), 'r') as json:
        json_file = json.read()
    model = model_from_json(json_file)
    model.load_weights(os.path.join('train_dir', model_name, 'weights.h5'))
    pred = model.predict(test_x).reshape(-1)
    results[model_name] = pred.tolist()
    
# Save the predictions
df = pd.DataFrame.from_dict(results)
df.to_csv("prediction_coreset.csv", index=None)

# Calculate the correlations
pearson_corr = df.corr(method='pearson').iloc[1:, 0]
spearman_corr = df.corr(method='spearman').iloc[1:, 0]
kendall_corr = df.corr(method='kendall').iloc[1:, 0]
corr_df = pd.DataFrame([pearson_corr, spearman_corr, kendall_corr], index=['pearson', 'spearman', 'kendall']).transpose()
corr_df.to_csv("correlations_coreset.csv", index=None)
