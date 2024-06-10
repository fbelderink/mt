from preprocessing.dictionary import Dictionary
import sys
sys.path.append("../")
from utils.file_manipulation import load_data
from assignments.assignment3 import *
from torch.utils.data import DataLoader
from preprocessing.dataset import TranslationDataset
from multiprocessing import freeze_support
import torch
from model.basic_net import BasicNet
from utils.ConfigLoader import ConfigLoader
from utils.hyperparameters import Hyperparameters
from functools import reduce
import operator
import numpy as np
class Score:
    def __init__(self, source_dict_path, target_dict_path, source_text_path, target_text_path, window_size, batch_size, config_path, number_of_sentences = 600 ,shuffle = False):
        self.source_dict : Dictionary = Dictionary.load(source_dict_path)
        self.target_dict : Dictionary= Dictionary.load(target_dict_path)
        self.batch_size : int = batch_size
        self.probList = [[]] * 600
        source_text = load_data(source_text_path)
        target_text = load_data(target_text_path)
        dict_de = Dictionary.load(source_dict_path)
        dict_en = Dictionary.load(target_dict_path)
        generate_dataset(source_text, target_text, window_size, 7000, dict_de = dict_de, dict_en = dict_en,save_path='data/scoring-set.pt')
        scoring_set : TranslationDataset = TranslationDataset.load("data/scoring-set.pt")
        self.scoring_dataloader = DataLoader(scoring_set, batch_size=batch_size, shuffle=shuffle)
        #batches ready
        self.eos_index = self.target_dict.get_index_of_string("</s>")
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        config = Hyperparameters(ConfigLoader("../configs/config.yaml").get_config())
        self.model = BasicNet(len(self.source_dict), len(self.target_dict), config,
                     window_size=window_size).to(self.device)
        if config.saved_model != "":
            self.model.load_state_dict(torch.load(config.saved_model))
    def get_scores(self):
        overflow = False
        last_sprint = False
        current_sentence_index = 0
        for S,T,L in self.scoring_dataloader:
            S = S.to(self.device)
            T = T.to(self.device)
            L = L.long().to(self.device)
            # L == self.eos_index : sto get a new tensor, where 1 at index of vector L if value at that index == self.eos, else 0
            #then get non zero indices
            indices = torch.nonzero(L == self.eos_index)
            indices = indices[:, 0].tolist()
            if indices[-1] != self.batch_size:
                overflow = True
            result = self.model(S, T)
            label_list = L.view(-1).tolist()
            for i, item in enumerate(label_list):
                prob_of_item = result[i, item].item()

                self.probList[current_sentence_index].append(np.log(prob_of_item))
                if not last_sprint:
                    if i == indices[0]:
                        indices = indices[1:]
                        if indices == []:
                            last_sprint = True
                        current_sentence_index += 1
            last_sprint = False
            print(self.probList)
            quit()
        sums = []
        for problist in self.probList:
            sums.append((sum(problist)))
        return self.probList


test = Score("../eval/dict_de.pkl",
             "../eval/dict_en.pkl",
             "../data/600s.de",
             "../data/600s.en",
             2,
             200,
             "../configs/config.yaml")

print(test.get_scores())