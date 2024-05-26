import torch.nn as nn
from model.layers.linear import LinearLayer
import torch
import math
import torch.nn.functional as F
from utils.hyperparameters import Hyperparameters


class BasicNet(nn.Module):
    def __init__(self,batch_size,source_dict_size,target_dict_size, config: Hyperparameters):
        super(BasicNet, self).__init__()
        """
        TODO: 
        - replace linear layers optionally
        - eigener Linear layer verwenden
        - add Batch norm
        """
        #Hyper parameters


        self.hidden_dim_1 = config.dimensions[0]
        self.hidden_dim_2 = config.dimensions[1]
        self.hidden_dim_3 = 2*self.hiddem_dim_2
        self.embed_dim = config.dimensions[2]

        self.activation_function = config.activation
        self.optimizer = config.optimizer
        self.init_learning_rate = config.learning_rate
        self.use_custom_ll = config.use_custom_ll
        self.batch_size = config.batch_size

        self.loss_sum = nn.CrossEntropyLoss(reduction="sum")
        self.loss_mean = nn.CrossEntropyLoss()
        self.replace_linear_layer = not (config.use_custom_ll)

        # embedding layers 
        self.source_embedding = nn.Embedding(source_dict_size, self.embed_dim)
        self.target_embedding = nn.Embedding(target_dict_size, self.embed_dim)

        if self.replace_linear_layer:
            # Fully connected source and target layers
            self.fc_source = nn.Linear(self.embed_dim, self.hidden_dim_1)
            self.fc_target = nn.Linear(self.embed_dim, self.hidden_dim_1)

            # Concatenation layer
            self.concat = nn.Linear(self.hidden_dim_1, self.hidden_dim_2)

            # Fully Connected Layer 1
            self.fc1 = nn.Linear(self.hidden_dim_2, self.hidden_dim_2)

            # Fully Connected Layer 2 / Projection
            self.fc2 = nn.Linear(self.hidden_dim_2, self.hidden_dim_3)

            # Output layer
            self.output_layer = nn.Linear(self.hidden_dim_3,target_dict_size)
        else:
            # Fully connected source and target layers
            self.fc_source = LinearLayer(batch_size,self.embed_dim, self.hidden_dim_1)
            self.fc_target = LinearLayer(batch_size,self.embed_dim, self.hidden_dim_1)

            # Concatenation layer
            self.concat = LinearLayer(batch_size,self.hidden_dim_1, self.hidden_dim_2)

            # Fully Connected Layer 1
            self.fc1 = LinearLayer(batch_size,self.hidden_dim_2, self.hidden_dim_2)

            # Fully Connected Layer 2 / Projection
            self.fc2 = LinearLayer(batch_size,self.hidden_dim_2, self.hidden_dim_2)

            # Output layer
            self.output_layer = LinearLayer(batch_size,self.hidden_dim_2,target_dict_size)





    def forward(self, S, T):

        # embedding
        src_embedded = self.source_embedding(S)
        tgt_embedded = self.target_embedding(T)

        # Fully connected layers
        src_fc = self.fc_source(src_embedded)
        src_fc = self.activation_function(src_fc)
        tgt_fc = self.fc_target(tgt_embedded)
        tgt_fc = self.activation_function(tgt_fc)

        # Concatenate source and target representations
        # join at feature dimension
        concat = self.concat(torch.cat((src_fc, tgt_fc), dim=1))
        concat = self.activation_function(concat)

        # Fully connected layers
        fc1_output = self.fc1(concat)
        fc1_output = self.activation_function(fc1_output)

        fc2_output = self.fc2(fc1_output)
        fc2_output = self.activation_function(fc2_output)

        # Output layer with softmax activation
        output = self.output_layer(fc2_output)

        # reshape output so that it has dimensions batch_size x target_voc_size
        output = output.mean(dim=1)

        return output




    def compute_loss(self, pred, label, normalized = True):
        label = label.view(-1)
        if normalized:
            loss = self.loss_mean(pred, label)
        else:
            loss = self.loss_sum(pred, label)
        return loss







    def print_structure(self):
        print("total number of trainable parameters: " + str(sum(p.numel() for p in self.parameters() if p.requires_grad)))
        print("Layers:")
        for name, module in self.named_children():
            if not isinstance(module, nn.CrossEntropyLoss):
                print(f"{name}: {module}")







# just for testing
def _custom_cross(pred,label):
    loss = 0
    count = 0
    for p, l in zip(pred, label):
        count+=1
        print("values: " + str(p[l]))

        loss += -math.log(p[l])

    return loss/count