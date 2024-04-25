import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoFeatureExtractor, AutoModel, HubertModel, Wav2Vec2FeatureExtractor
import soundfile as sf
from dotenv import load_dotenv


class RegressionHead(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        # self.linearLayer = nn.Linear(hidden_size, hidden_size)
        # self.out_proj = nn.Linear(hidden_size, 1)
        # self.seq = nn.Sequential(
        #       nn.Linear(hidden_size, hidden_size),
        #       # nn.ReLU(),
        #       # nn.Linear(hidden_size, hidden_size),
        #       # nn.ReLU(),
        #       # nn.Linear(hidden_size, hidden_size),
        #       # nn.ReLU(),
        #       # nn.Linear(hidden_size, hidden_size),
        #       nn.ReLU(),
        #       nn.Linear(hidden_size, hidden_size),
        #       nn.ReLU(),
        #       nn.Linear(hidden_size, 1))

        self.seq = nn.Sequential(

              nn.Linear(hidden_size, hidden_size),

              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, hidden_size * 2),
              nn.ReLU(),
              nn.Linear(hidden_size * 2, hidden_size * 4),
              nn.ReLU(),
              nn.Linear(hidden_size * 4, hidden_size * 2),
              nn.ReLU(),
              nn.Linear(hidden_size * 2, hidden_size),
              nn.ReLU(),
              nn.Linear(hidden_size, 300),
              nn.ReLU(),

              nn.Linear(300, 100),
              nn.ReLU(),

              nn.Linear(100, 10),
              nn.ReLU(),
              nn.Linear(10 , 1))

        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(hidden_size, 1)

    def forward(self, features):
        x = self.seq(features)
        # return x

        # x = features
        # x = self.dropout(x)
        # x = self.dense(x)
        # x = torch.tanh(x)
        # x = self.dropout(x)
        # x = self.out_proj(x)
        return x
    

# Model class
class Dysarthria_model(nn.Module):
  def __init__(self, model_path = "facebook/hubert-base-ls960", pooling_mode = "mean"):
        super().__init__()
        # self.processor = AutoFeatureExtractor.from_pretrained(model_path)
        self.SSLModel = AutoModel.from_pretrained(model_path)
        self.RegressionHead = RegressionHead(768)
        self.pooling_mode = pooling_mode

  def freeze_feature_extractor(self):
        self.SSLModel.feature_extractor._freeze_parameters()

  def freeze_whole_SSL_model(self):
        self.SSLModel._freeze_parameters()

  def merged_strategy(self, output_features, mode="mean"):
        if mode == "mean":
            outputs = torch.mean(output_features, dim=1)
        elif mode == "sum":
            outputs = torch.sum(output_features, dim=1)
        elif mode == "max":
            outputs = torch.max(output_features, dim=1)[0]
        else:
            raise Exception(
                "The pooling method hasn't been defined! Your pooling mode must be one of these ['mean', 'sum', 'max']")

        return outputs

  def forward(self, x):

    output_features = self.SSLModel(x).last_hidden_state
    hidden_states = self.merged_strategy(output_features, mode=self.pooling_mode)
    output = self.RegressionHead(hidden_states)

    return output


