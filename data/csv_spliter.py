import pandas as pd
from sklearn.model_selection import train_test_split
import sys


def spliter(path):
	# Load CSV
	df = pd.read_csv(path, header=None)

	# Get features and labels
	X = df.iloc[:, 2:]
	y = df.iloc[:, 1]
	 
	# Split the dataset
	X_train, X_test, y_train, y_test = train_test_split(
	    X, y, test_size=0.3, random_state=0)
	print(f'train size {len(X_train)} test size {len(X_test)}')

	# Save train and test files
	pd.concat([pd.concat([y_train, X_train], axis=1)]).to_csv('train.csv', header=None)
	pd.concat([pd.concat([y_test, X_test], axis=1)]).to_csv('test.csv', header=None)

if __name__ == '__main__':
	spliter(sys.argv[1])
