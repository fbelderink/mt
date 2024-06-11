
from search import beam_search
from search import greedy_search
import torch
from preprocessing.dictionary import Dictionary
from typing import List
from utils.file_manipulation import load_data
from preprocessing.postprocessing import undo_prepocessing
from metrics import BLEU


def get_bleu_of_model(MODEL_PATH: str,
                        source_data_path: str,
                        refs_path: str,
                        source_dict_path: str,
                        target_dict_path: str,
                        beam_size: int,
                        window_size: int,
                        do_beam_search: bool):
    
    model = torch.load(MODEL_PATH)
    source_data = load_data(source_data_path)
    refs = load_data(refs_path)
    source_dict = Dictionary.load(source_dict_path)
    target_dict = Dictionary.load(target_dict_path)
    
    translation = None
    if do_beam_search:
        translation = beam_search.translate(model,source_data,source_dict,target_dict,beam_size,window_size)
    else:
        translation = greedy_search.translate(model,source_data,source_dict,target_dict,window_size)
    
    translation = undo_prepocessing(translation)

    bleu = BLEU()
    bleu_score = bleu(translation, refs)


    return bleu_score 



#print(get_bleu_of_model("eval/checkpoints/10-06-2024/17_16_39.pth", "data/multi30k.dev.de", "data/multi30k.dev.en","data/dicts/train_dict_de.pkl","data/dicts/train_dict_en.pkl",4,2,False))


