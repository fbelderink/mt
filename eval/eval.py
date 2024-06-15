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
import matplotlib.pyplot as plt
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

def evaluate_scores(checkpoint_path, source_data_path, reference_data_path, source_dict, reference_dict, beam_size, window_size, do_beam, save_path):
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
    file = open(save_path,"w")
    for entry in data_points:
        file.write(f" {entry} \n")
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

    eval_set: TranslationDataset = TranslationDataset.load(eval_data_set_path)
    eval_dataloader = DataLoader(eval_set, batch_size=200, shuffle=False, num_workers=4)
    step_counter = 0
    ppl_list = []

    for S,T,L in eval_dataloader:
        S = S.to(device)
        T = T.to(device)
        L = L.long().to(device)


        pred = model(S, T)

        loss = model.compute_loss(pred, L)

        loss.backward()

        # keep track of metrics
        step_counter +=1

        # print batch metrics
        batch_perplexity = float(torch.exp(loss))
        ppl_list.append(batch_perplexity)
        print(step_counter/len(eval_dataloader))
    print(sum(ppl_list)/len(ppl_list))
    return sum(ppl_list)/len(ppl_list)


def get_ppl_on_checkpoint(checkpoint_path: str, eval_data_set_path,
               source_dict: Dictionary, target_dict: Dictionary, window_size: int, save_path: str):
    model_file_paths = glob.glob(f"{checkpoint_path}/*.pth")
    model_file_paths.sort()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ppl_list = []
    for i,model_path in enumerate(model_file_paths):
        print(i)
        model = torch.load(model_path)
        ppl_list.append(getPPLonModel(model, eval_data_set_path,source_dict,target_dict,window_size))
    print(ppl_list)
    file = open(save_path, "w")
    for ppl in ppl_list:
        file.write(f"{ppl} \n")
    file.close()
    return ppl_list


def plotData(data_files_list: List[str], data_description: str,save_path: str):
    data_lists = []
    for i,file_path in enumerate(data_files_list):
        data = load_data(file_path)
        data = [float(x[0]) for x in data]
        data_lists.append(data)
    x_points = [x for x in range(1,len(data_lists[0])+1)]
    print(data_lists[0])
    plt.figure(figsize=(10, 6))
    colors = ["b","r","g"]
    for i, ydata in enumerate(data_lists):
        plt.plot(x_points, ydata,marker='o', linestyle='-', color=colors[i])
    plt.xlabel('Checkpoint')
    plt.ylabel(data_description)
    plt.grid(True)
    plt.savefig(save_path)


