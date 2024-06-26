# for imports append .. to path
import sys
sys.path.append('..')

from metrics.metrics import BLEU
from utils.file_manipulation import load_data
import torch
from utils.ConfigLoader import Hyperparameters
from preprocessing.dictionary import Dictionary
from preprocessing.BPE import generate_bpe
from statistics import mean
import argparse
from search.beam_search import translate as beam_translate
from search.greedy_search import translate as greedy_translate
from typing import List


def parse_args():
    parser = argparse.ArgumentParser(description='Translate a given source language to a target language.')
    parser.add_argument( '--model_path', type=str, help='The path to the model.')
    parser.add_argument( '--source_data_path', type=str, help='The path to the source data.')
    parser.add_argument( '--target_data_path', type=str, help='The path to the target data.')
    parser.add_argument('--source_dict_path', type=str, help='The path to the source dictionary.')
    parser.add_argument('--target_dict_path', type=str, help='The path to the target dictionary.')
    parser.add_argument('--config_path', type=str, help='The path to the configuration file.')
    parser.add_argument('--beam_search', type=bool, help='Whether to use beam search.')
    parser.add_argument('--window_size', type=int, help='The window size for beam search.')
    parser.add_argument('--out_hyps_path', type=str, help='The path to the output hypotheses.')
    return parser.parse_args()

def create_hyps(model_path: str,
                source_data_path: str,
                target_data_path: str,
                source_dict_path: str,
                target_dict_path: str,
                config_path: str,
                beam_search: bool):
    if beam_search:
        return beam_translate(torch.load(model_path),
                              load_data(source_data_path),
                              Dictionary.load(source_dict_path),
                              Dictionary.load(target_dict_path),
                       2,
                       2)

    else:
        return greedy_translate(model_path,
                         source_data_path,
                         target_data_path,
                         Dictionary.load(source_dict_path),
                         Dictionary.load(target_dict_path),
                         Hyperparameters(config_path))


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


def calculate_bleu(hyps: List[List[str]],
                   refs: List[List[str]],
                   beam_size: int) -> float:
    """
    Calculate the BLEU score between a reference and a hypothesis.
    Args: reference: str, the reference translation.
    hypothesis: str, the hypothesis translation.
    Returns: The BLEU score.
    """
    # flatten in dimension 0
    hyps = [h for hyp in hyps for h in hyp]
    refs = [r for r in refs for _ in range(beam_size)]
    bleu = BLEU()
    return bleu(refs, hyps)


def main():
    args = parse_args()
    hyps = create_hyps(args.model_path,
                       args.source_data_path,
                       args.target_data_path,
                       args.source_dict_path,
                       args.target_dict_path,
                       args.config_path,
                       args.beam_search)

    refs = load_data(args.target_data_path)

    # TODO: iterate over a dir of checkpoints and create a graph of BLEU scores
    print(
        calculate_bleu(hyps,
                       refs,
                2)
    )


if __name__ == "__main__":
    main()
