#!/usr/bin/env python
# coding: utf-8

# 
# - Basically a 1D CNN starter with bandpass. Filter size hard-coded from [https://www.kaggle.com/kit716/grav-wave-detection](https://www.kaggle.com/kit716/grav-wave-detection) which uses the simple architecture from https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.120.141103 
# - Added inference to @hidehisaarai1213 's PyTorch starter, iteration order changed from Y.Nakama's pipeline: "iter on loader first then load model" to "load model first then iter the loader"
# 

# ## Libraries

# In[26]:


import os
import time
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import scipy as sp
from scipy import signal
import tensorflow as tf  # for reading TFRecord Dataset
import tensorflow_datasets as tfds  # for making tf.data.Dataset to return numpy arrays
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from kaggle_datasets import KaggleDatasets
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from tqdm import tqdm
import os
from IPython.display import clear_output


# In[2]:


SAVEDIR = Path("./")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ## CFG

# In[20]:


class CFG:
    debug = False
    print_freq = 1000
    num_workers = 12
    scheduler = "CosineAnnealingLR"
    model_name = "1dcnn"
    epochs = 5
    T_max = 3
    lr = 0.01
    min_lr = 1e-6
    batch_size = 1024
    val_batch_size = 100
    weight_decay = 1e-7
    gradient_accumulation_steps = 1
    max_grad_norm = 1000
    seed = 42
    target_size = 1
    target_col = "target"
    n_fold = 5
    trn_fold = [0, 1, 2, 3]  # [0, 1, 2, 3, 4]
    train = True
    bandpass_params = dict(lf=20, 
                           hf=500)


# ## Utils

# In[4]:


# ====================================================
# Utils
# ====================================================
def get_score(y_true, y_pred):
    score = roc_auc_score(y_true, y_pred)
    return score


def init_logger(log_file=SAVEDIR / 'train.log'):
    from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=log_file)
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger

LOGGER = init_logger()


def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

seed_torch(seed=CFG.seed)


# ## TFRecord Loader
# 
# This is the heart of this notebook. Instead of using PyTorch's Dataset and DataLoader, here I define custom Loader that reads samples from TFRecords.
# 
# FYI, there's a library that does the same thing, but its implementation is not optimized, so it's slower.
# 
# https://github.com/vahidk/tfrecord

# In[5]:


gcs_paths = []
all_files = []
path = "./data1/train/"
n_trial = 0
#print(j)
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        #print(filename)
        #print(os.path.join(dirname, filename))
        gcs_path = os.path.join(dirname, filename)
        gcs_paths.append(gcs_path)
        print(gcs_path)
        all_files.extend(np.sort(np.array(tf.io.gfile.glob(gcs_path))))
            
    
print("train_files: ", len(all_files))
all_files = np.array(all_files)


# In[6]:



for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))


# In[7]:


def count_data_items(fileids, train=True):
    """
    Count the number of samples.
    Each of the TFRecord datasets is designed to contain 28000 samples for train
    22500 for test.
    """
    sizes = 28000 if train else 22500
    return len(fileids) * sizes


AUTO = tf.data.experimental.AUTOTUNE


# ## Bandpass
# 
# Modified from various notebooks and https://www.kaggle.com/c/g2net-gravitational-wave-detection/discussion/261721#1458564

# In[8]:


def bandpass(x, lf=20, hf=500, order=8, sr=2048):
    '''
    Cell 33 of https://www.gw-openscience.org/LVT151012data/LOSC_Event_tutorial_LVT151012.html
    https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html
    '''
    sos = signal.butter(order, [lf, hf], btype="bandpass", output="sos", fs=sr)
    normalization = np.sqrt((hf - lf) / (sr / 2))
    if x.ndim ==2:
        for i in range(3):
            x[i] = signal.sosfilt(sos, x[i]) * normalization
    elif x.ndim == 3: # batch
        for i in range(x.shape[0]):
            for j in range(3):
                x[i, j] = signal.sosfilt(sos, x[i, j]) * normalization
    return x


# In[9]:


def prepare_wave(wave):
    wave = tf.reshape(tf.io.decode_raw(wave, tf.float64), (3, 4096))
    normalized_waves = []
    scaling = tf.constant([1.5e-20, 1.5e-20, 0.5e-20], dtype=tf.float64)
    for i in range(3):
#         normalized_wave = wave[i] / tf.math.reduce_max(wave[i])
        normalized_wave = wave[i] / scaling[i]
        normalized_waves.append(normalized_wave)
    wave = tf.stack(normalized_waves, axis=0)
    wave = tf.cast(wave, tf.float32)
    return wave


def read_labeled_tfrecord(example):
    tfrec_format = {
        "wave": tf.io.FixedLenFeature([], tf.string),
        "wave_id": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.int64)
    }
    example = tf.io.parse_single_example(example, tfrec_format)
    return prepare_wave(example["wave"]), tf.reshape(tf.cast(example["target"], tf.float32), [1]), example["wave_id"]


def read_unlabeled_tfrecord(example, return_image_id):
    tfrec_format = {
        "wave": tf.io.FixedLenFeature([], tf.string),
        "wave_id": tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, tfrec_format)
    return prepare_wave(example["wave"]), example["wave_id"] if return_image_id else 0


def get_dataset(files, batch_size=16, repeat=False, cache=False, 
                shuffle=False, labeled=True, return_image_ids=True):
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTO, compression_type="GZIP")
    if cache:
        # You'll need around 15GB RAM if you'd like to cache val dataset, and 50~60GB RAM for train dataset.
        ds = ds.cache()

    if repeat:
        ds = ds.repeat()

    if shuffle:
        ds = ds.shuffle(1024 * 2)
        opt = tf.data.Options()
        opt.experimental_deterministic = False
        ds = ds.with_options(opt)

    if labeled:
        ds = ds.map(read_labeled_tfrecord, num_parallel_calls=AUTO)
    else:
        ds = ds.map(lambda example: read_unlabeled_tfrecord(example, return_image_ids), num_parallel_calls=AUTO)

    ds = ds.batch(batch_size)
    ds = ds.prefetch(AUTO)
    return tfds.as_numpy(ds)


# In[10]:


class TFRecordDataLoader:
    def __init__(self, files, batch_size=32, cache=False, train=True, 
                              repeat=False, shuffle=False, labeled=True, 
                              return_image_ids=True):
        self.ds = get_dataset(
            files, 
            batch_size=batch_size,
            cache=cache,
            repeat=repeat,
            shuffle=shuffle,
            labeled=labeled,
            return_image_ids=return_image_ids)
        
        self.num_examples = count_data_items(files, labeled)

        self.batch_size = batch_size
        self.labeled = labeled
        self.return_image_ids = return_image_ids
        self._iterator = None
    
    def __iter__(self):
        if self._iterator is None:
            self._iterator = iter(self.ds)
        else:
            self._reset()
        return self._iterator

    def _reset(self):
        self._iterator = iter(self.ds)

    def __next__(self):
        batch = next(self._iterator)
        return batch

    def __len__(self):
        n_batches = self.num_examples // self.batch_size
        if self.num_examples % self.batch_size == 0:
            return n_batches
        else:
            return n_batches + 1


# ## MODEL

# In[11]:


class CNN1d(nn.Module):
    """1D convolutional neural network. Classifier of the gravitational waves.
    Architecture from there https://journals.aps.org/prl/pdf/10.1103/PhysRevLett.120.141103
    """

    def __init__(self, debug=False):
        super().__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv1d(3, 64, kernel_size=64),
            nn.BatchNorm1d(64),
            nn.ELU(),
        )
        self.cnn2 = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=32),
            nn.AvgPool1d(kernel_size=8),
            nn.BatchNorm1d(64),
            nn.ELU(),
        )
        self.cnn3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=32),
            nn.BatchNorm1d(128),
            nn.ELU(),
        )
        self.cnn4 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=16),
            nn.AvgPool1d(kernel_size=6),
            nn.BatchNorm1d(128),
            nn.ELU(),
        )
        self.cnn5 = nn.Sequential(
            nn.Conv1d(128, 256, kernel_size=16),
            nn.BatchNorm1d(256),
            nn.ELU(),
        )
        self.cnn6 = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=16),
            nn.MaxPool1d(kernel_size=4),
            nn.BatchNorm1d(256),
            nn.ELU(),
        )
        self.fc1 = nn.Sequential(
            nn.Linear(256 * 11, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.4),
            nn.ELU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.4),
            nn.ELU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(64, 64),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Softplus(),
        )
        self.fc4 = nn.Sequential(
            nn.Linear(64, 1),
        )
        self.debug = debug

    def forward(self, x, pos=None):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.cnn5(x)
        x = self.cnn6(x)
        x = x.flatten(start_dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return x


# ## Helper functions

# In[12]:


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def max_memory_allocated():
    MB = 1024.0 * 1024.0
    mem = torch.cuda.max_memory_allocated() / MB
    return f"{mem:.0f} MB"


# ## Trainer

# In[28]:


def train_fn(files, model, criterion, optimizer, epoch, scheduler, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()

    # switch to train mode
    model.train()
    start = end = time.time()
    global_step = 0

    train_loader = TFRecordDataLoader(
        files, batch_size=CFG.batch_size, 
        shuffle=True)
    for step, d in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        x = bandpass(d[0], **CFG.bandpass_params)
        x = torch.from_numpy(x).to(device)
        labels = torch.from_numpy(d[1]).to(device)

        batch_size = labels.size(0)
        y_preds = model(x)
        loss = criterion(y_preds.view(-1), labels.view(-1))
        # record loss
        losses.update(loss.item(), batch_size)
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), CFG.max_grad_norm)
        if (step + 1) % CFG.gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        print('Epoch: [{0}/{1}][{2}/{3}] '
              'Loss: {loss.val:.4f}({loss.avg:.4f}) '
              'Grad: {grad_norm:.4f}  '
              'LR: {lr:.6f}  '
              'Elapsed: {remain:s} '
              'Max mem: {mem:s}'
              .format(
               epoch+1, CFG.epochs, step, len(train_loader),
               loss=losses,
               grad_norm=grad_norm,
               lr=scheduler.get_last_lr()[0],
               remain=timeSince(start, float(step + 1) / len(train_loader)),
               mem=max_memory_allocated()))
    return losses.avg


def valid_fn(files, model, criterion, device):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    scores = AverageMeter()
    # switch to evaluation mode
    model.eval()
    filenames = []
    targets = []
    preds = []
    start = end = time.time()
    valid_loader = TFRecordDataLoader(
        files, batch_size=CFG.batch_size * 2, shuffle=False)
    for step, d in enumerate(valid_loader):
        # measure data loading time
        data_time.update(time.time() - end)
        
        targets.extend(d[1].reshape(-1).tolist())
        filenames.extend([f.decode("UTF-8") for f in d[2]])
        x = bandpass(d[0], **CFG.bandpass_params)
        x = torch.from_numpy(x).to(device)
        labels = torch.from_numpy(d[1]).to(device)

        batch_size = labels.size(0)
        # compute loss
        with torch.no_grad():
            y_preds = model(x)
        loss = criterion(y_preds.view(-1), labels.view(-1))
        losses.update(loss.item(), batch_size)

        preds.append(y_preds.sigmoid().to('cpu').numpy())
        if CFG.gradient_accumulation_steps > 1:
            loss = loss / CFG.gradient_accumulation_steps
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        print('EVAL: [{0}/{1}] '
              'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
              'Elapsed {remain:s} '
              'Loss: {loss.val:.4f}({loss.avg:.4f}) '
              .format(
               step, len(valid_loader), batch_time=batch_time,
               data_time=data_time, loss=losses,
               remain=timeSince(start, float(step+1)/len(valid_loader)),
               ))
    predictions = np.concatenate(preds).reshape(-1)
    return losses.avg, predictions, np.array(targets), np.array(filenames)


# ## Train loop

# In[29]:


# ====================================================
# Train loop
# ====================================================
def train_loop(train_tfrecords: np.ndarray, val_tfrecords: np.ndarray, fold: int):
    
    LOGGER.info(f"========== fold: {fold} training ==========")
    
    # ====================================================
    # scheduler 
    # ====================================================
    def get_scheduler(optimizer):
        if CFG.scheduler=='ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                             mode='min', 
                                                             factor=CFG.factor, 
                                                             patience=CFG.patience, 
                                                             verbose=True, 
                                                             eps=CFG.eps)
        elif CFG.scheduler=='CosineAnnealingLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                             T_max=CFG.T_max, 
                                                             eta_min=CFG.min_lr, 
                                                             last_epoch=-1)
        elif CFG.scheduler=='CosineAnnealingWarmRestarts':
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 
                                                                       T_0=CFG.T_0, 
                                                                       T_mult=1, 
                                                                       eta_min=CFG.min_lr, 
                                                                       last_epoch=-1)
        return scheduler

    # ====================================================
    # model & optimizer
    # ====================================================
    model = CNN1d()
    model.to(device)
    model = nn.DataParallel(model)


    optimizer = optim.SGD(model.parameters(), lr=CFG.lr, weight_decay=CFG.weight_decay)
    scheduler = get_scheduler(optimizer)

    # ====================================================
    # loop
    # ====================================================
    criterion = nn.BCEWithLogitsLoss()

    best_score = 0.
    best_loss = np.inf
    
    for epoch in range(CFG.epochs):
        print("\n\n")
        start_time = time.time()
        
        # train
        avg_loss = train_fn(train_tfrecords, model, criterion, optimizer, epoch, scheduler, device)

        # eval
        avg_val_loss, preds, targets, files = valid_fn(val_tfrecords, model, criterion, device)
        valid_result_df = pd.DataFrame({"target": targets, "preds": preds, "id": files})
        
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(avg_val_loss)
        elif isinstance(scheduler, optim.lr_scheduler.CosineAnnealingLR):
            scheduler.step()
        elif isinstance(scheduler, optim.lr_scheduler.CosineAnnealingWarmRestarts):
            scheduler.step()

        # scoring
        score = get_score(targets, preds)

        elapsed = time.time() - start_time

        LOGGER.info(f'Epoch {epoch+1} - avg_train_loss: {avg_loss:.4f}  avg_val_loss: {avg_val_loss:.4f}  time: {elapsed:.0f}s')
        LOGGER.info(f'Epoch {epoch+1} - Score: {score:.4f}')

        if score > best_score:
            best_score = score
            LOGGER.info(f'Epoch {epoch+1} - Save Best Score: {best_score:.4f} Model')
            torch.save({'model': model.state_dict(), 
                        'preds': preds},
                        SAVEDIR / f'{CFG.model_name}_fold{fold}_best_score.pth')
        
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            LOGGER.info(f'Epoch {epoch+1} - Save Best Loss: {best_loss:.4f} Model')
            torch.save({'model': model.state_dict(), 
                        'preds': preds},
                        SAVEDIR / f'{CFG.model_name}_fold{fold}_best_loss.pth')
    
    valid_result_df["preds"] = torch.load(SAVEDIR / f"{CFG.model_name}_fold{fold}_best_loss.pth",
                                          map_location="cpu")["preds"]

    return valid_result_df


# In[30]:


torch.cuda.empty_cache()


# In[ ]:


def get_result(result_df):
    preds = result_df['preds'].values
    labels = result_df[CFG.target_col].values
    score = get_score(labels, preds)
    LOGGER.info(f'Score: {score:<.4f}')

if CFG.train:
    # train 
    oof_df = pd.DataFrame()
    kf = KFold(n_splits=CFG.n_fold, shuffle=True, random_state=CFG.seed)

    folds = list(kf.split(all_files))
    for fold in range(CFG.n_fold):
        if fold in CFG.trn_fold:
            trn_idx, val_idx = folds[fold]
            train_files = all_files[trn_idx]
            valid_files = all_files[val_idx]
            _oof_df = train_loop(train_files, valid_files, fold)
            oof_df = pd.concat([oof_df, _oof_df])
            LOGGER.info(f"========== fold: {fold} result ==========")
            get_result(_oof_df)
    # CV result
    LOGGER.info(f"========== CV ==========")
    get_result(oof_df)
    # save result
    oof_df.to_csv(SAVEDIR / 'oof_df.csv', index=False)


# ## Inference

# In[15]:


states = []
for fold  in CFG.trn_fold:
    states.append(torch.load(os.path.join(SAVEDIR, f'{CFG.model_name}_fold{fold}_best_score.pth')))


# In[16]:


gcs_paths = []
all_files = []
path = "./data1/test/"
n_trial = 0
#print(j)
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        #print(filename)
        #print(os.path.join(dirname, filename))
        gcs_path = os.path.join(dirname, filename)
        gcs_paths.append(gcs_path)
        print(gcs_path)
        all_files.extend(np.sort(np.array(tf.io.gfile.glob(gcs_path))))
    
print("test_files: ", len(all_files))
all_files = np.array(all_files)


# In[17]:


model= CNN1d()
model.to(device)

wave_ids = []
probs_all = []

for fold, state in enumerate(states):
    tqdm.write(f"\n\nFold{fold}")
    
    model.load_state_dict(state['model'])
    model.eval()
    probs = []

    test_loader = TFRecordDataLoader(all_files, batch_size=CFG.val_batch_size, 
                                     shuffle=False, labeled=False)

    for i, d in tqdm(enumerate(test_loader), total=len(test_loader)):
        x = bandpass(d[0], **CFG.bandpass_params)
        x = torch.from_numpy(x).to(device)

        with torch.no_grad():
            y_preds = model(x)
        preds = y_preds.sigmoid().to('cpu').numpy()
        probs.append(preds)

        if fold==0: # same test loader, no need to do this the second time
            wave_ids.append(d[1].astype('U13'))

    probs = np.concatenate(probs)
    probs_all.append(probs)

probs_avg = np.asarray(probs_all).mean(axis=0).flatten()
wave_ids = np.concatenate(wave_ids)


# In[18]:


test_df = pd.DataFrame({'id': wave_ids, 'target': probs_avg})
# Save test dataframe to disk
folds = '_'.join([str(s) for s in CFG.trn_fold])
test_df.to_csv(f'{CFG.model_name}_folds_{folds}.csv', index = False)

