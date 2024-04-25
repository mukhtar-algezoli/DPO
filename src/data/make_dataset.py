# -*- coding: utf-8 -*-
import click
import logging
import os
from pathlib import Path
import numpy as np
import pandas as pd
from dotenv import find_dotenv, load_dotenv
from torch.utils.data import Dataset
from transformers import AutoFeatureExtractor, AutoModel, HubertModel, Wav2Vec2FeatureExtractor
import soundfile as sf

from torch.utils.data import DataLoader



class UASpeechDataset(Dataset):
    def __init__(self, metadata_path, raw_data_dir, model_path, transform=None, target_transform=None):
        self.metadata = pd.read_csv(metadata_path)
        self.processor = AutoFeatureExtractor.from_pretrained(model_path)
        self.raw_data_dir  = raw_data_dir

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        audio_path = "./"+self.raw_data_dir +  self.metadata.loc[self.metadata.index[idx], 'path']
        audio_path = audio_path.replace("/content", "")
        speech, samplerate = sf.read(audio_path)
        # seconds = librosa.get_duration(path= audio_path)
        # print(seconds)
        speech = speech[0:350000]

        preprocessed_speech = self.processor(speech, padding="max_length",  max_length = 350000,return_tensors="pt", sampling_rate = samplerate).input_values
        # label = self.metadata.iloc[idx, 3]
        label = self.metadata.loc[self.metadata.index[idx], "Intelligibility_Percentage"]

        return preprocessed_speech, label


def get_train_test_val_set(args):
    # Load the data
    train_set = UASpeechDataset(os.path.join(args.data_splits_path, "train_df_reg3.csv"), args.data_path, args.SSL_model)
    test_set = UASpeechDataset(os.path.join(args.data_splits_path, "test_df_reg3.csv"), args.data_path, args.SSL_model)
    val_set = UASpeechDataset(os.path.join(args.data_splits_path, "val_df_reg3.csv"), args.data_path, args.SSL_model)

    train_dataloader = DataLoader(train_set, batch_size = args.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_set, batch_size = args.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size = args.batch_size, shuffle=True)


    return train_dataloader, test_dataloader, val_dataloader


