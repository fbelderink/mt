from utils.file_manipulation import load_data
from utils.file_manipulation import save_data
from preprocessing.dictionary import Dictionary 
import torch
from model.basic_net import BasicNet
from utils.hyperparameters import Hyperparameters
from utils.ConfigLoader import ConfigLoader
from preprocessing import fragment 
import numpy as np
from preprocessing.postprocessing import undo_prepocessing



def translate(model_path: str, source_data_path: str, target_data_path: str,
               source_dict_path: Dictionary, target_dict_path: Dictionary, config: Hyperparameters):

    # TODO change checkpointing so no hyperparameters need to be hardcoded
    window_size = 2

    # TODO
    alignment_factor = 1

    source_dict = Dictionary.load(source_dict_path)
    target_dict = Dictionary.load(target_dict_path)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BasicNet(len(source_dict), len(target_dict), config, window_size).to(device)
    model.load_state_dict(torch.load(model_path))

    model.eval()

    # preprocessed source sentences   
    source_data = source_dict.apply_vocabulary_to_text(load_data(source_data_path),bpe_performed=False)

    target_sentences = []

    for sentence in source_data:

        # contains all entries needed to translate current sentence
        S = fragment._get_source_window_matrix([sentence], [[0]*(len(sentence)*alignment_factor+1)], window_size)
        print(S)
        get_source_idx = np.vectorize(source_dict.get_index_of_string)
        S = torch.from_numpy(get_source_idx(S)) # replace tokens with corresponding indices

        # contains the entry needed to translate the next sentence 
        t_list = ['<s>'] * window_size

        get_target_idx = np.vectorize(target_dict.get_index_of_string)
        t_tensor = torch.from_numpy(get_target_idx(t_list)) # replace tokens with corresponding indices
        
        translated_sentence = []

        for s in S:
            if len(translated_sentence) > 0 and translated_sentence[-1] == '</s>':
                break

            print(t_list)

            s = s.unsqueeze(0)
            t_tensor = t_tensor.unsqueeze(0)

            s = s.to(device)
            t_tensor = t_tensor.to(device)


            pred = model(s, t_tensor)
            res = target_dict.get_string_at_index(int(torch.argmax(pred)))
            translated_sentence.append(res)
            
            # update t
            t_list.append(res)
            t_list.pop(0)
            t_tensor = torch.from_numpy(get_target_idx(t_list))
        
        target_sentences.append(translated_sentence)
        print(translated_sentence) 
    

    target_sentences = undo_prepocessing(target_sentences)
    save_data(target_data_path,target_sentences)









translate("eval/checkpoints/09-06-2024/18_01_21.pth","data_translation/source.txt","data_translation/target.txt","data/train_dict_de.pkl","data/train_dict_en.pkl",Hyperparameters(ConfigLoader("configs/config.yaml").get_config()))