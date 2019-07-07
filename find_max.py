from glob import glob
import json

results = dict()
with open('dicts.json', 'r') as f:
    results = json.load(f)

test = [res['pearson_test'] for k, res in results.items()]
valid = [res['pearson_valid'] for k, res in results.items()]
train = [res['pearson_train'] for k, res in results.items()]
test.sort(reverse=True)
train.sort(reverse=True)
valid.sort(reverse=True)

print("Total results: {}".format(len(results.keys())))
print("Maximum pearson in test: {:.2}, train: {:.2} and valid: {:.2}".format(max(test), max(train), max(valid)))
