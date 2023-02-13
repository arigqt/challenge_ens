import pandas as pd
import numpy as np
import os
import zipfile
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import sklearn.metrics

def get_data():
    datasets_dir = os.getcwd().replace('Code','Datasets')
    y_train_path = Path(datasets_dir + '\\y_train.csv')
    x_train_path = Path(datasets_dir + '\\X_train.zip')
    x_test_path = Path(datasets_dir + '\\X_test.zip')

    train_list = []
    with zipfile.ZipFile(x_train_path, 'r') as ziptrain:
        for info in ziptrain.infolist()[1:]:
            zip_img = ziptrain.open(info.filename)
            cv_img = cv2.imdecode(np.frombuffer(zip_img.read(), dtype=np.uint8),
                                cv2.IMREAD_GRAYSCALE)
            train_list.append(cv_img)
    X_train = np.stack(train_list, axis=0)
    print('#### X_train collected ####')

    test_list = []
    with zipfile.ZipFile(x_test_path, 'r') as ziptest:
        for info in ziptest.infolist()[1:]:
            zip_img = ziptest.open(info.filename)
            cv_img = cv2.imdecode(np.frombuffer(zip_img.read(), dtype=np.uint8),
                                cv2.IMREAD_GRAYSCALE)
            test_list.append(cv_img)        
    X_test = np.stack(test_list, axis=0)
    print('#### X_test collected ####')

    y_train = pd.read_csv(y_train_path, index_col=0).T
    print('#### y_train collected ####')

    return X_train, X_test, y_train

def prediction_to_df(df):
    return pd.DataFrame(np.stack(df, axis=0).reshape((len(df), -1)))

def rand_index_dataset(y_true_df: pd.DataFrame, y_pred_df: pd.DataFrame) -> float:
    """Compute the individual rand index for each sample in the dataset and then average it"""
    individual_rand_index = []
    for row_index in range(y_true_df.values.shape[0]):
        labels = y_true_df.values[row_index]
        preds = y_pred_df.values[row_index]
        individual_rand_index.append(sklearn.metrics.adjusted_rand_score(labels[labels != 0], preds[labels != 0]))

    return np.mean(individual_rand_index)

def plot_slice_seg(slice_image, seg):
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(slice_image, cmap="gray")
    axes[1].imshow(slice_image, cmap="gray")
    seg_masked = np.ma.masked_where(seg.reshape((512,512)) == 0, (seg.reshape((512,512))))
    axes[1].imshow(seg_masked, cmap="tab20")
    plt.axis("off")

def df_get_ith_image(df, i):
    return df.iloc[i].values.reshape((512,512))