import torch
from model.basic_net import BasicNet
from preprocessing.dictionary import Dictionary
from utils.hyperparameters import Hyperparameters
from utils.ConfigLoader import ConfigLoader

def load_checkpoint(checkpoint_dictionary_path: str, overwrite_config_path: str = None):

    #If wanted, specific hyperparameters can be overwritten when loading a checkpoint.
    #please note that changing structural parameters results in crashing


    source_dict = Dictionary.load("eval/dict_de.pkl")
    target_dict = Dictionary.load("eval/dict_en.pkl")
    config = Hyperparameters(ConfigLoader(f"{checkpoint_dictionary_path}/CONFIG.yml").get_config())
    if overwrite_config_path is not None:
        config = Hyperparameters(ConfigLoader(overwrite_config_path).get_config())

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = BasicNet(len(source_dict), len(target_dict), config, config.window_size).to(device)

    model.load_state_dict(torch.load(f"{checkpoint_dictionary_path}/MODEL.pth"))
    return model