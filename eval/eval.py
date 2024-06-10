# for imports append .. to path
import sys
sys.path.append('..')

from metrics.metrics import BLEU
from utils.file_manipulation import load_data, load_model
import torch
from model.basic_net import BasicNet
from preprocessing.dataset import TranslationDataset
from utils.ConfigLoader import Hyperparameters
from preprocessing.dictionary import Dictionary
from preprocessing.BPE import generate_bpe, perform_bpe


def get_data_dicts(data_path: str,
                   source_dict_path: str,
                   target_dict_path: str,
                   bpe_operations: int) -> (Dictionary, Dictionary):
    """
    Get the dictionaries for the source and target languages.
    Args:
        data_path:
        source_dict_path:
        target_dict_path:
        bpe_operations:

    Returns: The tuple of dictionaries.
    """
    if source_dict_path:
        source_dict = Dictionary.load(source_dict_path)
    else:
        source_data = load_data(data_path)
        operations = generate_bpe(source_data, bpe_operations)
        source_dict = Dictionary(source_data, operations)

    if target_dict_path:
        target_dict = Dictionary.load(target_dict_path)
    else:
        target_data = load_data(data_path)
        operations = generate_bpe(target_data, bpe_operations)
        target_dict = Dictionary(target_data, operations)

    return source_dict, target_dict


def create_hyps(model_path: str,
                data_path: str,
                source_dict_path: str,
                target_dict_path: str,
                hyp_path: str,
                config: Hyperparameters,
                bpe_operations: int = 100,
                window_size: int = 5,
                device: str = 'cpu',
                verbose: bool = False):
    """
    Create hypotheses using a model.
    Args:
    model_path: str, the path to the model.
    data_path: str, the path to the data.
    hyp_path: str, the path to save the hypotheses.
    device: str, the device to run the model on.
    verbose: bool, whether to print messages.
    """
    # set device, if wish can be fulfilled
    device = set_device(device)

    # load dictionaries for both source and target languages
    source_dict, target_dict = get_data_dicts(data_path,
                                              source_dict_path,
                                              target_dict_path,
                                              bpe_operations)

    # load the model
    model = BasicNet(source_dict.get_size(),
                     target_dict.get_size(),
                     config,
                     window_size).to(device)

    # load the weights of the checkpoint
    model = torch.load(model_path)

    # set model to evaluation mode
    model.eval()

    # load the data
    input_data = load_data(data_path)

    # apply BPE to the input data
    input_data = source_dict.apply_vocabulary_to_text(input_data)






def print_verbose(verbose: bool, message: str):
    """
    Print a message if verbose is set to True.
    Args:
    verbose: bool, whether to print the message.
    message: str, the message to print.
    """
    if verbose:
        print(message)


def set_device(device: str) -> torch.device:
    """
    Set the device to run the model on.
    Returns: The device to run the model on.
    """
    if device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif device == 'mps' and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    return device


def calculate_bleu(ref_path: str, hyp_path: str) -> float:
    """
    Calculate the BLEU score between a reference and a hypothesis.
    Args: reference: str, the reference translation.
    hypothesis: str, the hypothesis translation.
    Returns: The BLEU score.
    """
    reference = load_data(ref_path, split=False)
    hypothesis = load_data(hyp_path, split=False)
    bleu = BLEU()
    return bleu(reference, hypothesis)


def main():
    create_hyps(None, './data/val7k.pt', None, None, None )


if __name__ == "__main__":
    main()
