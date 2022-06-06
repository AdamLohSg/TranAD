import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
from src.folderconstants import *
from shutil import copyfile

datasets = ['synthetic', 'SMD', 'SWaT', 'SMAP', 'MSL', 'WADI', 'MSDS', 'UCR', 'MBA', 'NAB']

wadi_drop = ['2_LS_001_AL', '2_LS_002_AL','2_P_001_STATUS','2_P_002_STATUS']

def load_and_save(category, filename, dataset, dataset_folder):
    temp = np.genfromtxt(os.path.join(dataset_folder, category, filename),
                        dtype=np.float64,
                        delimiter=',')
    print(dataset, category, filename, temp.shape)
    np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)
    return temp.shape

def load_and_save2(category, filename, dataset, dataset_folder, shape):
    temp = np.zeros(shape)
    with open(os.path.join(dataset_folder, 'interpretation_label', filename), "r") as f:
        ls = f.readlines()
    for line in ls:
        pos, values = line.split(':')[0], line.split(':')[1].split(',')
        start, end, indx = int(pos.split('-')[0]), int(pos.split('-')[1]), [int(i)-1 for i in values]
        temp[start-1:end-1, indx] = 1
    print(dataset, category, filename, temp.shape)
    np.save(os.path.join(output_folder, f"SMD/{dataset}_{category}.npy"), temp)

def normalize(a):
    a = a / np.maximum(np.absolute(a.max(axis=0)), np.absolute(a.min(axis=0)))
    return (a / 2 + 0.5)

def normalize2(a, min_a = None, max_a = None):
    if min_a is None: min_a, max_a = min(a), max(a)
    return (a - min_a) / (max_a - min_a), min_a, max_a

def normalize3(a, min_a = None, max_a = None):
    if min_a is None: min_a, max_a = np.min(a, axis = 0), np.max(a, axis = 0)
    return (a - min_a) / (max_a - min_a + 0.0001), min_a, max_a

def convertNumpy(df):
    x = df[df.columns[3:]].values[::10, :]
    return (x - x.min(0)) / (x.ptp(0) + 1e-4)

def transformstr(df):
    for i in list(df): 
        df[i] = df[i].apply(lambda x: str(x).replace("," , "."))
    df = df.astype(float)
    return df

def load_data(dataset):
    folder = os.path.join(output_folder, dataset)
    os.makedirs(folder, exist_ok=True)
    if dataset == 'SWaT':
        dataset_folder = 'data/SWaT/'
        df_train = pd.read_csv(dataset_folder + "SWaT_Dataset_Normal_v1.csv")
        df_train = df_train.rename(columns=lambda x: x.strip())
        print("passed")
        df_train = df_train.drop(columns=['Timestamp', 'Normal/Attack'])
        df_test = pd.read_csv(dataset_folder + "SWaT_Dataset_Attack_v0.csv")
        df_test = df_test.rename(columns=lambda x: x.strip())
        labels = [ float(label!= 'Normal' ) for label  in df_test["Normal/Attack"].values]
        print("passed labels")
        df_test = df_test.drop(columns=['Timestamp', 'Normal/Attack'])
        print("passed df test and train")
        df_train, df_test = transformstr(df_train), transformstr(df_test)
        train, min_a, max_a = normalize3(df_train.values)
        test, _, _ = normalize3(df_test.values, min_a, max_a)
        size = train.shape[0]
        labels = np.broadcast_to(np.array(labels).reshape(-1, 1), test.shape)
        print(train.shape, test.shape, labels.shape)
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
    elif dataset == 'WADI':
        dataset_folder = 'data/WADI'
        ls = pd.read_csv(os.path.join(dataset_folder, 'WADI_attacklabels.csv'))
        train = pd.read_csv(os.path.join(dataset_folder, 'WADI_14days.csv'), skiprows=1000, nrows=2e5)
        test = pd.read_csv(os.path.join(dataset_folder, 'WADI_attackdata.csv'))
        train.dropna(how='all', inplace=True); test.dropna(how='all', inplace=True)
        train.fillna(0, inplace=True); test.fillna(0, inplace=True)
        test['Time'] = test['Time'].astype(str)
        test['Time'] = pd.to_datetime(test['Date'] + ' ' + test['Time'])
        labels = test.copy(deep = True)
        for i in test.columns.tolist()[3:]: labels[i] = 0
        for i in ['Start Time', 'End Time']: 
            ls[i] = ls[i].astype(str)
            ls[i] = pd.to_datetime(ls['Date'] + ' ' + ls[i])
        for index, row in ls.iterrows():
            to_match = row['Affected'].split(', ')
            matched = []
            for i in test.columns.tolist()[3:]:
                for tm in to_match:
                    if tm in i: 
                        matched.append(i); break			
            st, et = str(row['Start Time']), str(row['End Time'])
            labels.loc[(labels['Time'] >= st) & (labels['Time'] <= et), matched] = 1
        train, test, labels = convertNumpy(train), convertNumpy(test), convertNumpy(labels)
        print(train.shape, test.shape, labels.shape)
        for file in ['train', 'test', 'labels']:
            np.save(os.path.join(folder, f'{file}.npy'), eval(file))
    else:
        raise Exception(f'Not Implemented. Check one of {datasets}')

if __name__ == '__main__':
    commands = sys.argv[1:]
    load = []
    if len(commands) > 0:
        for d in commands:
            load_data(d)
    else:
        print("Usage: python preprocess.py <datasets>")
        print(f"where <datasets> is space separated list of {datasets}")