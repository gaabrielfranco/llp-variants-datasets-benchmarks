import numpy as np
import pandas as pd
import cv2

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

meta = unpickle("cifar-10-batches-py/batches.meta")
label_names = meta[b'label_names']

for i in range(1, 6):
    data = unpickle("cifar-10-batches-py/data_batch_" + str(i))
    if i == 1:
        X_train = data[b'data']
        y_train = data[b'labels']
    else:
        X_train = np.concatenate((X_train, data[b'data']))
        y_train = np.concatenate((y_train, data[b'labels']))

test_data = unpickle("cifar-10-batches-py/test_batch")
X_test = test_data[b'data']
y_test = np.array(test_data[b'labels'])

labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

X_train = X_train.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")

X_test = X_test.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype("uint8")

X_train = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_train])
X_test = np.array([cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in X_test])

# Flatten images
X_train = X_train.reshape(-1, 32*32)
X_test = X_test.reshape(-1, 32*32)

# Create dataframe
df_train = pd.DataFrame(X_train)
df_train['label'] = y_train

df_test = pd.DataFrame(X_test)
df_test['label'] = y_test

df = pd.concat([df_train, df_test])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df = df.rename(columns={i: str(i) for i in range(1024)})

df.to_parquet("cifar-10-grey.parquet")
print("File saved to cifar-10-grey.parquet")