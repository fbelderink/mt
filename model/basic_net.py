import torch.nn as nn
from model.layers.linear import LinearLayer
import torch
import math
from utils.hyperparameters import Hyperparameters


class BasicNet(nn.Module):
    def __init__(self, source_dict_size, target_dict_size, config: Hyperparameters, window_size):
        super(BasicNet, self).__init__()

        # hyperparameters
        self.embed_dim = config.dimensions[0]
        self.hidden_dim_1 = config.dimensions[1]
        self.hidden_dim_2 = config.dimensions[2]

        self.window_size = window_size

        self.activation_function = config.activation

        self.use_custom_ll = config.use_custom_ll

        # LOSS
        self.loss_sum = nn.CrossEntropyLoss(reduction="sum")
        self.loss_mean = nn.CrossEntropyLoss()

        # LAYERS
        # embedding layers
        self.source_embedding = nn.Embedding(source_dict_size, self.embed_dim)
        self.target_embedding = nn.Embedding(target_dict_size, self.embed_dim)

        # Fully connected source and target layers
        self.fc_source = nn.Linear(self.embed_dim * (2 * self.window_size + 1), self.hidden_dim_1)
        self.fc_source_bn = nn.BatchNorm1d(self.hidden_dim_1)
        self.dropout_source = nn.Dropout(p=config.dropout_rate)

        self.fc_target = nn.Linear(self.embed_dim * self.window_size, self.hidden_dim_1)
        self.fc_target_bn = nn.BatchNorm1d(self.hidden_dim_1)
        self.dropout_target = nn.Dropout(p=config.dropout_rate)

        # Fully Connected Layer 1
        self.fc1 = nn.Linear(2 * self.hidden_dim_1, self.hidden_dim_2)
        self.fc1_bn = nn.BatchNorm1d(self.hidden_dim_2)
        self.dropout1 = nn.Dropout(p=config.dropout_rate)

        # Fully Connected Layer 2 / Projection
        self.fc2 = nn.Linear(self.hidden_dim_2, target_dict_size)
        self.fc2_bn = nn.BatchNorm1d(target_dict_size)

        if self.use_custom_ll:
            # Fully connected source and target layers
            self.fc_source = LinearLayer(self.embed_dim * (2 * self.window_size + 1), self.hidden_dim_1)
            self.fc_target = LinearLayer(self.embed_dim * self.window_size, self.hidden_dim_1)

            # Fully Connected Layer 1
            self.fc1 = LinearLayer(2 * self.hidden_dim_1, self.hidden_dim_2)

            # Fully Connected Layer 2 / Projection
            self.fc2 = LinearLayer(self.hidden_dim_2, target_dict_size)

    def forward(self, S, T, apply_log_softmax=True):

        # embedding
        src_embedded = self.source_embedding(S)
        tgt_embedded = self.target_embedding(T)

        src_embedded = torch.flatten(src_embedded, start_dim=1)
        tgt_embedded = torch.flatten(tgt_embedded, start_dim=1)

        # Fully connected layers
        src_fc = self.fc_source(src_embedded)
        src_fc = self.fc_source_bn(src_fc)
        src_fc = self.activation_function(src_fc)
        src_fc = self.dropout_source(src_fc)

        tgt_fc = self.fc_target(tgt_embedded)
        tgt_fc = self.fc_target_bn(tgt_fc)
        tgt_fc = self.activation_function(tgt_fc)
        tgt_fc = self.dropout_target(tgt_fc)

        # Concatenate source and target representations
        # join at feature dimension
        concat = torch.cat((src_fc, tgt_fc), dim=1)
        concat = self.activation_function(concat)

        # Fully connected layers
        fc1_output = self.fc1(concat)
        fc1_output = self.fc1_bn(fc1_output)
        fc1_output = self.activation_function(fc1_output)
        fc1_output = self.dropout1(fc1_output)

        fc2_output = self.fc2(fc1_output)
        fc2_output = self.fc2_bn(fc2_output)

        # Output layer with softmax activation in nn.CrossEntropyLoss
        output = fc2_output

        if not self.training and apply_log_softmax:
            output = torch.nn.functional.log_softmax(output, dim=1)

        return output

    def compute_loss(self, pred, label, normalized=True):
        label = label.view(-1)
        if normalized:
            loss = self.loss_mean(pred, label)
        else:
            loss = self.loss_sum(pred, label)
        return loss

    def print_structure(self):
        print("total number of trainable parameters: " + str(
            sum(p.numel() for p in self.parameters() if p.requires_grad)))
        print("Layers:")
        for name, module in self.named_children():
            if not isinstance(module, nn.CrossEntropyLoss):
                print(f"{name}: {module}")
