import random

import torch

from utils.hyperparameters import Hyperparameters
from utils.ConfigLoader import ConfigLoader
from training.train import train
import glob
from utils.file_manipulation import *
from metrics.calculate_bleu_of_model import get_bleu_of_model
from search import beam_search
from scoring import score
from preprocessing.fragment import fragment_data_to_indices
from preprocessing.dictionary import Dictionary
from preprocessing.dataset import TranslationDataset
from torch.utils.data import DataLoader
def getseeds():
    seeds = []
    for i in range(3):
        seeds.append(random.randint(-(1e10),1e10))
    if seeds[0] == seeds[1] or seeds[0] == seeds[2] or seeds[1] == seeds[2]:
        return getseeds()
    else:
        return seeds


def exectute_runs():
    for name, seed in zip(["A", "B", "C"], getseeds()):
        train("data/train7k-w3.pt", None, Hyperparameters(ConfigLoader("configs/best_config.yaml").get_config()), max_epochs=5,
          shuffle=True, num_workers=4, val_rate=100, train_eval_rate=10, random_seed=seed, model_name = name, save_ppl = True)

def evaluate_scores(checkpoint_path, source_data_path, reference_data_path, source_dict, reference_dict, beam_size, window_size, do_beam):
    source_data = load_data(source_data_path)
    reference_data = load_data(reference_data_path)
    #ppl_file_paths = glob.glob(f"{checkpoint_path}/*.ppl")
    model_file_paths = glob.glob(f"{checkpoint_path}/*.pth")
   # ppl_file_paths.sort()
    model_file_paths.sort()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    data_points =[]
    for i,model_path in enumerate(model_file_paths):
        '''ppl_file = open(ppl_path,"r")
        ppl = ppl_file.read()
        ppl_file.close()'''
        print(i)
        model = torch.load(model_path, map_location=device)
        bleu_score = get_bleu_of_model(model, source_data, reference_data, source_dict, reference_dict, beam_size, window_size, do_beam)


        data_points.append(bleu_score)
    print(data_points)
    file = open("plot.data","w")
    for entry in data_points:
        file.write(entry+" \n")
    file.close()
    return data_points


def getScoresForEval():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load("eval/checkpoints/best_model/15_01_58.pth", map_location=device)
    source_data = load_data("data/raw/multi30k.dev.de")
    target_data =  load_data("data/raw/multi30k.dev.en")
    source_dict = Dictionary.load("eval/dict_de.pkl")
    target_dict = Dictionary.load("eval/dict_en.pkl")
    beam_size = 3
    window_size = 3
    our_translations = beam_search.translate(model, source_data, source_dict, target_dict, beam_size, window_size)
    reference_score = score.get_scores(model, source_data, target_data,source_dict, target_dict,window_size)
    our_score = score.get_scores(model, source_data, our_translations,source_dict, target_dict,window_size)
    print(sum(our_score)/len(our_score))
    print(sum(reference_score)/len(reference_score))


def getPPLonModel(model: nn.Module, eval_data_set_path,
               source_dict: Dictionary, target_dict: Dictionary, window_size: int):


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()
    ppl_list = []
    counter = 0
    train_set: TranslationDataset = TranslationDataset.load(eval_data_set_path)
    train_dataloader = DataLoader(train_set, batch_size=200, shuffle=False, num_workers=4)
    for S,T,L in train_dataloader:
        print(counter)
        counter += 1
        S, T, L = fragment_data_to_indices([source_sentence], [target_sentence],
                                           window_size, source_dict, target_dict)

        S = torch.from_numpy(S)
        T = torch.from_numpy(T)
        L = torch.from_numpy(L).long()



        #call model without log softmax to be able to compute loss
        pred = model(S, T, False)
        loss = model.compute_loss(pred, L, False)
        print(torch.exp(loss).item())
        ppl_list.append(torch.exp(loss).item())

    #return average of ppl
    print(sum(ppl_list)/len(ppl_list))
    return sum(ppl_list)/len(ppl_list)



