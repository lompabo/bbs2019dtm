#/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import numpy as np
from matplotlib import pyplot as plt
import os
import pandas as pd

def load_lr_data(fname):
    '''
    Load data for the language recognition problem
    '''
    # parse datafile
    snippets = []
    languages = []
    with open(os.path.join('resources', fname)) as f:
    # parse all lines in the data file
        for l in f:
            # ul = l.decode('utf-8')
            ul = l
            # get fields separated by commas
            txt, language = ul.split("@")
            # trim strings
            txt = txt.strip()
            language = language.strip()
            # build a snippet object for ease of access
            snippets.append(txt)
            languages.append(language)
    return snippets, languages


def plot_confusion_matrix(targets, preds, classes):
    '''
    Plot a confusion matrix with fancy colors
    '''
    n = len(classes)
    # Build the confusion matrix
    cm = np.zeros((n, n))
    for t, p in zip(targets, preds):
        cm[t,p] += 1
    # Display the confusion matrix
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True class')
    plt.xlabel('Predicted class')
    plt.show()


def standardize_data(data_train, data_val=None, data_test=None):
    '''
    Standardize data
    '''
    # We loop over all columns
    data_train_cols = [] # A list for the normalized columns
    data_test_cols = [] # A list for the normalized columns
    data_val_cols = [] # A list for the normalized columns
    for j in range(data_train.shape[1]):
        mu = np.mean(data_train[:, j])
        sigma = np.std(data_train[:, j])
        # If the standard deviation is 0, it means that the feature is
        # constant over all the training set. A feature such as this is
        # useless for training
        if sigma != 0:
            data_train_cols.append((data_train[:, j] - mu) / sigma)
            if data_val is not None:
                data_val_cols.append((data_test[:, j] - mu) / sigma)
            if data_test is not None:
                data_test_cols.append((data_test[:, j] - mu) / sigma)
        else:
            print('Feature #%d is constant and has been removed' % j)
    # Convert the list of columns again to numpy arrays
    # NOTE: vstack builds an array by stacking
    res = (np.vstack(data_train_cols).transpose(), )
    if data_val is not None:
        res = res + (np.vstack(data_test_cols).transpose(), )
    if data_test is not None:
        res = res + (np.vstack(x_test_cols).transpose(), )
    return res


def de_standardize(data, means, stds):
    res = np.full(data.shape, np.nan)
    for j in range(data.shape[1]):
        res[:, j] = (data[:, j] * stds[j]) + means[j]
    return res


def labels_to_int(data, labels):
    '''
    Convert generic labels to integer
    '''
    # NOTE: shouldn't I use pandas categories for this
    res = []
    for label in data:
        # NOTE: "index" returns the index of an item in list
        res.append(labels.index(label))
    return np.array(res)


def colored_histogram(xdata, ydata, *args, **xargs):
    '''
    Histogram + class colors
    '''
    # Separate the input data according to the class
    tmp = []
    yvalues = set(ydata)
    for l in yvalues:
        tmp.append(xdata[ydata == l])
    plt.figure()
    plt.hist(tmp, *args, stacked=True, **xargs)
    if 'label' in xargs: plt.legend()
    plt.show()


def load_stock_data(fname):
    # parse datafile
    data = pd.read_csv(fname)
    # Convert dates to python datetime objects and reindex
    dates = pd.to_datetime(data.Date)
    data.set_index(dates, inplace=True)
    data.drop('Date', 1, inplace=True)
    # Obtain the day of the year as an integer
    yday = np.array([d.timetuple().tm_yday for d in dates])
    yday = pd.Series(index=data.index, data=yday)
    data['yday'] = yday
    # closing = np.array(data.Close)
    return data
    # return yday, closing, np.array(dates)


def load_air_data():
    # parse datafile
    fname =  os.path.join('resources', 'AirQualityUCI.csv')
    data = pd.read_csv(fname, sep=';')
    # Convert dates to python datetime objects and reindex
    dates = pd.to_datetime(data.Date, format='%d/%m/%Y-%H.%M.%S')
    data.set_index(dates, inplace=True)
    data.drop('Date', 1, inplace=True)
    data = data.replace(-200, np.nan)
    # Replace "jolly" values with nans
    # data.loc[:, 'CO(GT)'] = data['CO(GT)'].replace(-200, np.nan)
    # for col in data.columns:
    #     data.loc[:, col] = data[col].replace(-200, np.nan)
    # Obtain the day of the year as an integer
    # yday = np.array([d.timetuple().tm_yday for d in dates])
    # yday = pd.Series(index=data.index, data=yday)
    # data['yday'] = yday
    # closing = np.array(data.Close)
    return data
    # return yday, closing, np.array(dates)


def load_device_data():
    # parse datafile
    fname = os.path.join('resources', 'ElectricDevices1.csv')
    data = pd.read_csv(fname, header=None)
    return data


# def sliding_win_ds(data, targets, nsteps_in, nsteps_out=1):
#     nsamples = len(data) - nsteps_in - nsteps_out + 1
#     ncols = len(data.columns)
#     x = np.full((nsamples, nsteps_in, ncols), np.nan)
#     y = np.full((nsamples, nsteps_out * len(targets)), np.nan)
#     for i in range(nsteps_in, nsteps_in + nsamples):
#         x[i-nsteps_in, :, :] = data.iloc[i-nsteps_in:i].values
#         vals = data[targets].iloc[i:i+nsteps_out].values
#         y[i-nsteps_in, :] = vals.reshape((nsteps_out * len(targets), ))
#     index = data.index[np.arange(nsteps_in, nsteps_in + nsamples)]
#     return index, x, y


def sliding_win_ds(data, targets, nsteps_in, nsteps_out=1, stepsize=1):
    nsamples = len(data) - nsteps_in - nsteps_out + 1
    totwidth = nsteps_in + nsteps_out
    x = np.stack( (data[i:1+i-totwidth or None:stepsize] for i in range(0, nsteps_in)), axis=1)
    y = np.stack( (data[i:1+i-totwidth or None:stepsize][targets] for i in range(nsteps_in, nsteps_in+nsteps_out)), axis=1)
    index = data.index[np.arange(nsteps_in, nsteps_in + nsamples)]
    return index, x, y


def persistence_baseline(data, targets, nsteps_in, nsteps_out=1, stepsize=1):
    nsamples = len(data) - nsteps_in - nsteps_out + 1
    totwidth = nsteps_in + nsteps_out
    y = np.stack( (data[nsteps_in-1:nsteps_in-totwidth or None:stepsize][targets] for i in range(nsteps_in, nsteps_in+nsteps_out)), axis=1)
    index = data.index[np.arange(nsteps_in, nsteps_in + nsamples)]
    return y


# def persistence_baseline(data, targets, nsteps_in, nsteps_out=1):
#     nsamples = len(data) - nsteps_in - nsteps_out + 1
#     ncols = len(data.columns)
#     y = np.full((nsamples, nsteps_out * len(targets)), np.nan)
#     for i in range(nsteps_in, nsteps_in + nsamples):
#         vals = data[targets].iloc[i-1].values
#         y[i-nsteps_in, :] = np.repeat(vals, nsteps_out)
#     return y

def persistence_diff_baseline(data, targets, nsteps_in, nsteps_out=1):
    nsamples = len(data) - nsteps_in - nsteps_out + 1
    ncols = len(data.columns)
    y = np.full((nsamples, nsteps_out * len(targets)), np.nan)
    for i in range(nsteps_in, nsteps_in + nsamples):
        pre2 = data[targets].iloc[i-2].values
        pre1 = data[targets].iloc[i-1].values
        vals = pre1 + (pre1 - pre2)
        y[i-nsteps_in, :] = np.repeat(vals, nsteps_out)
    return y

# def sliding_win_ds(data, targets, nsteps_in):
#     nsamples = len(data) - nsteps_in
#     # infields = [c for c in data.columns if c != target]
#     ncols = len(data.columns)
#     x = np.full((nsamples, nsteps_in, ncols), np.nan)
#     y = np.full((nsamples, len(targets)), np.nan)
#     for i in range(nsteps_in, nsteps_in + nsamples):
#         x[i-nsteps_in, :, :] = data.iloc[i-nsteps_in:i].values
#         y[i-nsteps_in, :] = data[targets].iloc[i:i+1].values
#     index = data.index[np.arange(nsteps_in, nsteps_in + nsamples)]
#     return index, x, y


def sliding_win_pred(model, data_start, nsteps_pred):
    data = data_start
    nsteps_in = data.shape[0]
    preds = []
    for i in range(nsteps_pred):
        pred = model.predict(np.array([data]))
        data = np.vstack([data[1:, :], pred])
        preds.append(pred[0])
    return np.array(preds)


def pred_real_scatter(real, pred, title):
    plt.figure(figsize=(10, 5))
    plt.scatter(real, pred, c=np.abs(real-pred), cmap='RdYlGn_r', linewidths=0)
    plt.title(title)
    plt.xlabel('Prediction')
    plt.xlabel('Target')
    plt.show()


def pred_real_plot(index, real, pred, title):
    plt.figure(figsize=(10, 5))
    plt.plot(index, pred, label='pred')
    plt.plot(index, real, label='real')
    plt.title(title)
    plt.legend()
    plt.show()


def get_r2(real, pred):
    ssres = np.sum((real - pred)**2)
    mu = np.mean(real)
    sstot = np.sum((real - mu)**2)
    return 1 - ssres / sstot


def get_mae(real, pred):
    return np.mean(np.abs(real - pred))


def get_mse(real, pred):
    return np.mean((real - pred)**2)

def get_mare(real, pred):
    return np.mean(np.abs((real - pred) / real))


def low_pass(series, nsteps):
    kernel = np.full((nsteps,), 1.0/nsteps)
    newdata = np.convolve(series.values, kernel, mode='valid')
    if nsteps > 2:
        newindex = series.index[nsteps//2:-nsteps//2+1]
    else:
        newindex = series.index[1:]
    return pd.Series(index=newindex, data=newdata)

def normalize_mu_sigma(x_train, x_val=None, x_test=None):
    # We loop over all columns
    x_train_cols = [] # A list for the normalized columns
    x_test_cols = [] # A list for the normalized columns
    x_val_cols = [] # A list for the normalized columns
    for j in range(x_train.shape[1]):
        mu = np.mean(x_train[:, j])
        sigma = np.std(x_train[:, j])
        # If the standard deviation is 0, it means that the feature is
        # constant over all the training set. A feature such as this is
        # useless for training
        if sigma != 0:
            x_train_cols.append((x_train[:, j] - mu) / sigma)
            if x_val is not None:
                x_val_cols.append((x_test[:, j] - mu) / sigma)
            if x_test is not None:
                x_test_cols.append((x_test[:, j] - mu) / sigma)
        else:
            print('Feature #%d is constant over the set and has been removed' % j)
    # Convert the list of columns again to numpy arrays
    # NOTE: vstack builds an array by stacking
    res = (np.vstack(x_train_cols).transpose(), )
    if x_val is not None:
        res = res + (np.vstack(x_test_cols).transpose(), )
    if x_test is not None:
        res = res + (np.vstack(x_test_cols).transpose(), )
    return res

