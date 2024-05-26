import torch.nn as nn
from model.layers.linear import LinearLayer
import torch
import math
import torch.nn.functional as F



class BasicNet(nn.Module):
    def __init__(self,batch_size,source_dict_size,target_dict_size, replace_linear_layer=True):
        super(BasicNet, self).__init__()
        """
        TODO: 
        - add Batch norm
        """
        
        #Hyper parameters
        embed_size = 100
        hidden_dim_1 = 500 
        hidden_dim_2 = 500
        hidden_dim_3 = 2*hidden_dim_2

        # Model's loss function
        self.loss_sum = nn.CrossEntropyLoss(reduction="sum")
        self.loss_mean = nn.CrossEntropyLoss()

        # batch Norm
        """
        self.bn_fc_target = nn.BatchNorm1d(hidden_dim_1)
        self.bn_concat = nn.BatchNorm1d(hidden_dim_2)
        self.bn_fc1 = nn.BatchNorm1d(hidden_dim_2)
        self.bn_fc2 = nn.BatchNorm1d(hidden_dim_3)
        self.bn_fc_source = nn.BatchNorm1d(hidden_dim_1)
        """

        # embedding layers 
        self.source_embedding = nn.Embedding(source_dict_size, embed_size)
        self.target_embedding = nn.Embedding(target_dict_size, embed_size)

        if replace_linear_layer:
            # Fully connected source and target layers
            self.fc_source = nn.Linear(embed_size, hidden_dim_1)
            self.fc_target = nn.Linear(embed_size, hidden_dim_1)

            # Concatenation layer 
            self.concat = nn.Linear(hidden_dim_1, hidden_dim_2)

            # Fully Connected Layer 1
            self.fc1 = nn.Linear(hidden_dim_2, hidden_dim_2)
            
            # Fully Connected Layer 2 / Projection
            self.fc2 = nn.Linear(hidden_dim_2, hidden_dim_3)

            # Output layer
            self.output_layer = nn.Linear(hidden_dim_3,target_dict_size)
        else:
            # Fully connected source and target layers
            self.fc_source = LinearLayer(batch_size,embed_size, hidden_dim_1)
            self.fc_target = LinearLayer(batch_size,embed_size, hidden_dim_1)

            # Concatenation layer 
            self.concat = LinearLayer(batch_size,hidden_dim_1, hidden_dim_2)

            # Fully Connected Layer 1
            self.fc1 = LinearLayer(batch_size,hidden_dim_2, hidden_dim_2)
            
            # Fully Connected Layer 2 / Projection
            self.fc2 = LinearLayer(batch_size,hidden_dim_2, hidden_dim_2)

            # Output layer
            self.output_layer = LinearLayer(batch_size,hidden_dim_2,target_dict_size)
        




    def forward(self, S, T):

        # embedding
        src_embedded = self.source_embedding(S)
        tgt_embedded = self.target_embedding(T)

        # Fully connected layers
        src_fc = self.fc_source(src_embedded)
        src_fc = F.relu(src_fc)
        tgt_fc = self.fc_target(tgt_embedded)
        tgt_fc = F.relu(tgt_fc)
        
        # Concatenate source and target representations
        # join at feature dimension
        concat = self.concat(torch.cat((src_fc, tgt_fc), dim=1))
        concat = F.relu(concat)

        # Fully connected layers
        fc1_output = self.fc1(concat)
        fc1_output = F.relu(fc1_output)

        fc2_output = self.fc2(fc1_output)
        fc2_output = F.relu(fc2_output)
        
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