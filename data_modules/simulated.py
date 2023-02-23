# Methods for loading and parsing the simulated version of the ascat dataset into a dataframe.
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn import preprocessing
from sklearn.model_selection import train_test_split as sk_split
import pytorch_lightning as pl
import os
from abc import ABC
import pandas as pd
import numpy as np
import logging
import random


class SimulateSinusoidal:
    """
    Create or load simulated version of ASCAT count number data.
    """

    @staticmethod
    def damped_signal(t, initial_amplitude, decay_rate, angular_frequency, init_phase_angle):
        return initial_amplitude * np.exp(-decay_rate * t) * np.cos(angular_frequency * t - init_phase_angle)

    @staticmethod
    def make_singularity(x, t1, t2, amplitude=1):
        """
        Make singularity structure in a batch of signals
        x: batched signal (Batch size, signal length)
        """
        x[:, t1:t2] += amplitude
        return x

    def make_transient(self, x, initial_amplitude, decay_rate, angular_frequency, init_phase_angle, t1, t2):
        """
        Make a transient structure in a batch of signals
        x: batched signal (Batch size, signal length)
        """
        sig_t = np.arange(t1, t2)
        x[:, sig_t] += self.damped_signal(sig_t / t2, initial_amplitude, decay_rate, angular_frequency, init_phase_angle)
        return x

    @property
    def frame(self):
        # TODO: makes new frame every call
        # This is what is read into the loader
        d = {'features': [self.signals[i, :] for i in range(self.n)],
             'labels': self.labels,
             'sub_labels': self.sub_labels
             }
        return pd.DataFrame(data=d)

    def __init__(self, classes=2, samples=2000, channels=2, sig_length=100, save_to_file=None):

        self.classes = classes
        self.n = samples
        self.channels = channels
        self.length = sig_length
        self.t = np.arange(self.length)

        # And randomly assign each sample to a different class with equal probability
        self.labels = np.random.choice([i for i in range(classes)], size=self.n, p=[1./classes for _ in range(classes)])
        self.sub_labels = np.random.choice([i for i in range(4)], size=self.n, p=[1./4 for _ in range(4)])
        self.signals = np.zeros((self.n, self.channels, self.length))

        # Shared parameters (a class will take a random choice from this set)
        num_shared = 5
        # base signal
        base_af_shared = np.random.normal(5.5, 2, num_shared)
        # singularity
        singularity_width = int(np.floor(self.length / 40))
        singularity_t1 = [random.randint(0, self.length - singularity_width) for _ in range(num_shared)]

        for cls in range(self.classes):

            # base signal parameters
            base_angular_freq = [np.max((2.5, np.random.choice(base_af_shared))) * np.pi for _ in range(self.channels)]
            base_amplitude = [np.max((0.5, np.random.normal(1, 0.3, 1))) for _ in range(self.channels)]
            base_init_phase_angle = 0

            # singularity parameters
            s_t1 = np.random.choice(singularity_t1)
            s_t2 = s_t1 + singularity_width
            s_amplitude = [np.max((0.2, np.random.normal(1, 0.3, 1))) for _ in range(self.channels)]

            mask = (np.ma.getmask(np.ma.masked_equal(self.labels, cls))) #& \
                       # (np.ma.getmask(np.ma.masked_equal(self.sub_labels, sub_cls)))

            # Add base signal
            for channel in range(self.channels):
                self.signals[mask, channel, :] = \
                    base_amplitude[channel] * np.cos((base_angular_freq[channel] * self.t / self.length)
                                                     - base_init_phase_angle)

            # Add singularity signal
            for channel in range(self.channels):
                self.signals[mask, channel, :] = self.make_singularity(self.signals[mask, channel, :],
                                                                       s_t1, s_t2,
                                                                       amplitude=s_amplitude[channel])

            if save_to_file is not None:
                # save data to file for benchmarking on same seed
                self.frame.to_csv(save_to_file, index=False)

        self.signals += np.random.normal(0, 0.2, self.signals.shape)

    def __str__(self):
        s = "SimulateSinusoidal class summary\n==========================="
        s += "TODO"
        return s


class SinusoidalDataModule(SimulateSinusoidal, pl.LightningDataModule, ABC):
    """
    """
    @property
    def num_cancer_types(self):
        return len(self.label_encoder.classes_)

    def __init__(self, classes=2, samples=1000, channels=2, sig_length=100, batch_size=128, save_to_file=None):
        """
        @param steps:               Number of steps to Markov Chain
        @param classes:             Number of different Markov Chains
        @param length:              Dimension of each state in Markov Chain
        @param n:                   Number of samples
        @param n_kernels_per:       Number of kernel transitions per Markov Chain
        @param n_kernels_shared:    Number of those kernel transitions that are shared between chains
        @param path:                Path for saving
        @param batch_size:          Batch size to load data into model
        """

        self.batch_size = batch_size
        self.training_set, self.test_set, self.validation_set = None, None, None

        # Define simulated set, and run process forward
        SimulateSinusoidal.__init__(self, classes=classes, samples=samples, channels=channels, sig_length=sig_length,
                                    save_to_file=save_to_file)

        _df = self.frame

        # Encode remaining type labels, so they can be used by the model later
        self.label_encoder = preprocessing.LabelEncoder()
        self.label_encoder.fit_transform(_df.labels.unique())

        # Split frame into training, validation, and test
        self.train_df, test_df = sk_split(_df, test_size=0.2)
        self.test_df, self.val_df = sk_split(test_df, test_size=0.2)

        self.setup()

    def setup(self, stage=None):
        self.training_set = SinusoidalDataset(self.train_df, self.label_encoder)
        self.test_set = SinusoidalDataset(self.test_df, self.label_encoder)
        self.validation_set = SinusoidalDataset(self.val_df, self.label_encoder)

    def train_dataloader(self):
        return DataLoader(
            sampler=None,
            dataset=self.training_set,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            shuffle=True
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validation_set,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            shuffle=False
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_set,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            shuffle=False
        )


class SinusoidalDataset(Dataset):

    def __init__(self, data: pd.DataFrame, label_encoder):
        """
        """
        self.data_frame = data
        self.label_encoder = label_encoder
        self.n = len(self.data_frame.index)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Get features
        sample = self.data_frame.loc[self.data_frame.index[idx]]
        feature = sample['features']

        # Get label
        label = self.data_frame.loc[self.data_frame.index[idx], ['labels']][0]
        label_enc = list(self.label_encoder.classes_).index(label)
        # Get sub-label
        sub_label = self.data_frame.loc[self.data_frame.index[idx], ['sub_labels']][0]

        batch = {"feature": torch.tensor(feature, dtype=torch.float),
                 "label": torch.tensor(label_enc),
                 # "sub_label": torch.tensor(sub_label)
                 }
        return batch
