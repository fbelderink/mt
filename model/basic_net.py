import torch.nn as nn
from model.layers import linear
import torch
import math
import torch.nn.functional as F



class BasicNet(nn.Module):
    def __init__(self,batch_size,source_dict_size,target_dict_size, replace_linear_layer=False):
        super(BasicNet, self).__init__()
        """
        TODO: 
        - replace linear layers optionally
        - eigener Linear layer verwenden
        - add Batch norm
        """
        #Hyper parameters
        embed_dim = 200
        hidden_dim_1 = 300 
        hidden_dim_2 = 500

        # embedding layers 
        self.source_embedding = nn.Embedding(source_dict_size, embed_dim)
        self.target_embedding = nn.Embedding(target_dict_size, embed_dim)

        # Fully connected source and target layers
        self.fc_source = nn.Linear(embed_dim, hidden_dim_1)
        self.fc_target = nn.Linear(embed_dim, hidden_dim_1)

        # Concatenation layer 
        self.concat = nn.Linear(hidden_dim_1, hidden_dim_2)

        # Fully Connected Layer 1
        self.fc1 = nn.Linear(hidden_dim_2, target_dict_size)
        
        # Fully Connected Layer 2 / Projection
        self.fc2 = nn.Linear(target_dict_size, target_dict_size)

        # Output layer
        self.output_layer = nn.Linear(target_dict_size,target_dict_size)

        # Model's loss function
        self.criterion = nn.CrossEntropyLoss()

        #self.linear1 = linear.LinearLayer(batch_size,5,5,True)




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
        # TODO might be wrong 
        output = output.mean(dim=1)

        return output




    def compute_loss(self, pred, label):
        label = label.view(-1)
        loss = self.criterion(pred, label)

        return loss






    def print_structure(self):
        """
        TODO:
        vor Beginn des Trainings die Gesamtzahl der trainierbaren Parameter sowie die ver-
        wendeten shapes von allen verwendeten Tensoren (also die Dimensionalität deren Aus-
        gabe) zusammen mit dem entsprechenden Namen der Layer auszugeben und damit
        die Netzwerkstruktur deutlich machen.

        """

        for i in self.named_children():
            print(i)
        print()






# just for testing
def _custom_cross(pred,label):
    loss = 0
    count = 0
    for p, l in zip(pred, label):
        count+=1
        print("values: " + str(p[l]))

        loss += -math.log(p[l])

    return loss/count