import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
import collections
import numbers
import math
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
from pathlib import Path


class PSMSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train", train_start=0.0, train_end=1.0, return_index: bool = False):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        self.return_index = return_index
        data = pd.read_csv(data_path + '/train.csv')
        data = data.values[:, 1:]

        data = np.nan_to_num(data)

        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = pd.read_csv(data_path + '/test.csv')

        test_data = test_data.values[:, 1:]
        test_data = np.nan_to_num(test_data)

        self.test = self.scaler.transform(test_data)

        start = int(len(data) * train_start)
        end = int(len(data) * train_end)
        self.train = data[start:end]
        self.val = self.test

        self.test_labels = pd.read_csv(data_path + '/test_label.csv').values[:, 1:]

        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):
        """
        Number of images in the object dataset.
        """
        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        start = index * self.step
        if self.mode == "train":
            window = self.train[start:start + self.win_size]
            label = self.test_labels[0:self.win_size]
        elif self.mode == 'val':
            window = self.val[start:start + self.win_size]
            label = self.test_labels[0:self.win_size]
        elif self.mode == 'test':
            window = self.test[start:start + self.win_size]
            label = self.test_labels[start:start + self.win_size]
        else:
            start = index * self.win_size
            window = self.test[start:start + self.win_size]
            label = self.test_labels[start:start + self.win_size]
        if self.return_index:
            return np.float32(window), np.float32(label), start
        return np.float32(window), np.float32(label)


class MSLSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train", train_start=0.0, train_end=1.0, return_index: bool = False):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        self.return_index = return_index
        data = np.load(data_path + "/MSL_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/MSL_test.npy")
        self.test = self.scaler.transform(test_data)
        start = int(len(data) * train_start)
        end = int(len(data) * train_end)
        self.train = data[start:end]
        self.val = self.test
        self.test_labels = np.load(data_path + "/MSL_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        start = index * self.step
        if self.mode == "train":
            window = self.train[start:start + self.win_size]
            label = self.test_labels[0:self.win_size]
        elif self.mode == 'val':
            window = self.val[start:start + self.win_size]
            label = self.test_labels[0:self.win_size]
        elif self.mode == 'test':
            window = self.test[start:start + self.win_size]
            label = self.test_labels[start:start + self.win_size]
        else:
            start = index * self.win_size
            window = self.test[start:start + self.win_size]
            label = self.test_labels[start:start + self.win_size]
        if self.return_index:
            return np.float32(window), np.float32(label), start
        return np.float32(window), np.float32(label)


class SMAPSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train", train_start=0.0, train_end=1.0, return_index: bool = False):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        self.return_index = return_index
        data = np.load(data_path + "/SMAP_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMAP_test.npy")
        self.test = self.scaler.transform(test_data)
        start = int(len(data) * train_start)
        end = int(len(data) * train_end)
        self.train = data[start:end]
        self.val = self.test
        self.test_labels = np.load(data_path + "/SMAP_test_label.npy")
        print("test:", self.test.shape)
        print("train:", self.train.shape)

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        start = index * self.step
        if self.mode == "train":
            window = self.train[start:start + self.win_size]
            label = self.test_labels[0:self.win_size]
        elif self.mode == 'val':
            window = self.val[start:start + self.win_size]
            label = self.test_labels[0:self.win_size]
        elif self.mode == 'test':
            window = self.test[start:start + self.win_size]
            label = self.test_labels[start:start + self.win_size]
        else:
            start = index * self.win_size
            window = self.test[start:start + self.win_size]
            label = self.test_labels[start:start + self.win_size]
        if self.return_index:
            return np.float32(window), np.float32(label), start
        return np.float32(window), np.float32(label)


class SMDSegLoader(object):
    def __init__(self, data_path, win_size, step, mode="train", train_start=0.0, train_end=1.0, return_index: bool = False):
        self.mode = mode
        self.step = step
        self.win_size = win_size
        self.scaler = StandardScaler()
        self.return_index = return_index
        data = np.load(data_path + "/SMD_train.npy")
        self.scaler.fit(data)
        data = self.scaler.transform(data)
        test_data = np.load(data_path + "/SMD_test.npy")
        self.test = self.scaler.transform(test_data)
        start = int(len(data) * train_start)
        end = int(len(data) * train_end)
        self.train = data[start:end]
        data_len = len(self.train)
        self.val = self.train[(int)(data_len * 0.8):]
        self.test_labels = np.load(data_path + "/SMD_test_label.npy")

    def __len__(self):

        if self.mode == "train":
            return (self.train.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'val'):
            return (self.val.shape[0] - self.win_size) // self.step + 1
        elif (self.mode == 'test'):
            return (self.test.shape[0] - self.win_size) // self.step + 1
        else:
            return (self.test.shape[0] - self.win_size) // self.win_size + 1

    def __getitem__(self, index):
        start = index * self.step
        if self.mode == "train":
            window = self.train[start:start + self.win_size]
            label = self.test_labels[0:self.win_size]
        elif self.mode == 'val':
            window = self.val[start:start + self.win_size]
            label = self.test_labels[0:self.win_size]
        elif self.mode == 'test':
            window = self.test[start:start + self.win_size]
            label = self.test_labels[start:start + self.win_size]
        else:
            start = index * self.win_size
            window = self.test[start:start + self.win_size]
            label = self.test_labels[start:start + self.win_size]
        if self.return_index:
            return np.float32(window), np.float32(label), start
        return np.float32(window), np.float32(label)


def _skab_autodetect_cols(df: pd.DataFrame):
    ts_col = None
    for c in df.columns:
        if np.issubdtype(df[c].dtype, np.datetime64):
            ts_col = c
            break
        try:
            pd.to_datetime(df[c])
            ts_col = c
            break
        except Exception:
            pass

    label_col = None
    for c in df.columns:
        uniq = pd.unique(df[c].dropna())
        if len(uniq) <= 3 and set(pd.Series(uniq).dropna().astype(str)).issubset({"0", "1"}):
            label_col = c
            break

    feats = [c for c in df.columns if c not in {ts_col, label_col}]
    feats = [c for c in feats if np.issubdtype(df[c].dtype, np.number)]
    return ts_col, label_col, feats


def _skab_read_csv(path: Path):
    df = pd.read_csv(path)
    ts_col, label_col, feats = _skab_autodetect_cols(df)
    if ts_col is None:
        raise ValueError(f"[SKAB] timestamp column not found in {path}")
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.sort_values(ts_col).reset_index(drop=True)
    df = df.ffill()
    if label_col is None:
        df["__label__"] = 0
        label_col = "__label__"
    return df, ts_col, label_col, feats


def build_skab_dataset(
    data_root: str,
    window_size: int,
    stride: int,
    split_mode: str = "scenario",
    train_ratio: float = 0.6,
    val_ratio: float = 0.1,
    test_ratio: float = 0.3,
):
    root = Path(data_root)
    groups = ["anomaly-free", "valve1", "valve2", "other"]
    files = []
    for g in groups:
        files += list((root / g).glob("*.csv"))
    files = sorted(files, key=lambda p: (p.parent.name, p.name))

    n = len(files)
    n_train = int(round(n * train_ratio))
    n_val = int(round(n * val_ratio))
    train_files = files[:n_train]
    val_files = files[n_train : n_train + n_val]
    test_files = files[n_train + n_val :]

    def make_windows(file_list):
        X, y = [], []
        for fp in file_list:
            df, ts_col, label_col, feats = _skab_read_csv(fp)
            vals = df[feats].to_numpy(dtype=np.float32)
            lbls = df[label_col].to_numpy(dtype=np.int64)
            T = len(df)
            for s in range(0, max(1, T - window_size + 1), stride):
                e = s + window_size
                if e > T:
                    break
                X.append(vals[s:e])
                y.append(int(lbls[s:e].max()))
        X = np.stack(X) if len(X) else np.zeros((0, window_size, 1), dtype=np.float32)
        y = np.array(y, dtype=np.int64)
        return X, y

    X_tr, y_tr = make_windows(train_files)
    mu = (
        X_tr.mean(axis=(0, 1), keepdims=True)
        if X_tr.size
        else np.array(0.0, dtype=np.float32)
    )
    sigma = (
        X_tr.std(axis=(0, 1), keepdims=True) + 1e-8
        if X_tr.size
        else np.array(1.0, dtype=np.float32)
    )
    X_tr = (X_tr - mu) / sigma if X_tr.size else X_tr
    X_va, y_va = make_windows(val_files)
    if X_va.size:
        X_va = (X_va - mu) / sigma
    X_te, y_te = make_windows(test_files)
    if X_te.size:
        X_te = (X_te - mu) / sigma

    meta = {
        "n_files_total": n,
        "n_train": len(train_files),
        "n_val": len(val_files),
        "n_test": len(test_files),
        "feature_dim": int(X_tr.shape[-1]) if X_tr.size else 0,
        "window_size": window_size,
        "stride": stride,
        "split_mode": split_mode,
        "files_train": [str(p) for p in train_files],
        "files_val": [str(p) for p in val_files],
        "files_test": [str(p) for p in test_files],
        "mu": np.asarray(mu).squeeze().tolist() if np.size(mu) else [],
        "sigma": np.asarray(sigma).squeeze().tolist() if np.size(sigma) else [],
    }
    return (X_tr, y_tr), (X_va, y_va), (X_te, y_te), meta


class _SKABDataset(Dataset):
    def __init__(self, X, y, return_index: bool = False):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
        self.return_index = return_index

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.return_index:
            return self.X[idx], self.y[idx], idx
        return self.X[idx], self.y[idx]


_SKAB_CACHE = None


def get_loader_segment(data_path, batch_size, win_size=100, step=100, mode='train', dataset='KDD', train_start=0.0, train_end=1.0, return_index: bool = False, return_meta: bool = False, **kwargs):
    if dataset.upper() == 'SKAB':
        global _SKAB_CACHE
        if _SKAB_CACHE is None:
            _SKAB_CACHE = build_skab_dataset(
                data_root=data_path,
                window_size=win_size,
                stride=step,
            )
        (Xtr, ytr), (Xva, yva), (Xte, yte), meta = _SKAB_CACHE
        split_map = {'train': (Xtr, ytr), 'val': (Xva, yva), 'test': (Xte, yte), 'thre': (Xte, yte)}
        X, y = split_map.get(mode, (Xtr, ytr))
        dataset_obj = _SKABDataset(X, y, return_index=return_index)
        shuffle = mode == 'train'
        loader = DataLoader(dataset_obj, batch_size=batch_size, shuffle=shuffle, drop_last=False)
        if return_meta:
            meta = {"input_c": meta.get("feature_dim"), "output_c": meta.get("feature_dim"), **meta}
            return loader, meta
        return loader

    if (dataset == 'SMD'):
        dataset = SMDSegLoader(data_path, win_size, step, mode, train_start, train_end, return_index=return_index)
    elif (dataset == 'MSL'):
        dataset = MSLSegLoader(data_path, win_size, 1, mode, train_start, train_end, return_index=return_index)
    elif (dataset == 'SMAP'):
        dataset = SMAPSegLoader(data_path, win_size, 1, mode, train_start, train_end, return_index=return_index)
    elif (dataset == 'PSM'):
        dataset = PSMSegLoader(data_path, win_size, 1, mode, train_start, train_end, return_index=return_index)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             num_workers=0)
    if return_meta:
        return data_loader, {}
    return data_loader
