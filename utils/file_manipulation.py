from typing import List, Union
import torch
import torch.nn as nn


def load_data(path: str,
              split: bool = True) -> Union[List[str], List[List[str]]]:
    """
    Load data from a file.
    Args:
        path: The path to the file.
        split: Whether to split the data by whitespace.
    Returns: List of strings if split=False or list of lists of strings otherwise.

    """
    in_file = open(path, 'r', encoding='utf-8')

    if split:
        data = [line.split() for line in in_file]
    else:
        data = [line for line in in_file]
    in_file.close()

    return data


def save_data(path: str, data: List[List[str]]):
    out_file = open(path, 'w', encoding='utf-8')

    for sentence in data:
        line = ' '.join(sentence)
        out_file.write(line + '\n')

    out_file.close()


def save_batches(path: str, batches: List[tuple]):
    out_file = open(path, 'w', encoding='utf-8')
    for idx, batch in enumerate(batches):
        S, T, L = batch
        assert S.shape[0] == T.shape[0] == L.shape[0]
        out_file.write(f"Batch {idx + 1}\n")
        for i in range(S.shape[0]):
            line = f"S: {S[i, :]}, T: {T[i, :]}, L: {L[i]}\n"
            out_file.write(line)

    out_file.close()


def save_model(path: str, model: nn.Module):
    torch.save(model.state_dict(), path)


def load_model(path: str, model: nn.Module) -> nn.Module:
    model.load_state_dict(torch.load(path))
    return model

def save_checkpoint():
    pass