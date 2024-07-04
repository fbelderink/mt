import argparse
import glob
from utils.file_manipulation import *
from metrics.calculate_bleu_of_model import get_bleu_of_model
from search import beam_search
from scoring import score
from preprocessing.dictionary import Dictionary


def get_bleu_of_checkpoints(checkpoint_path: str,
                            source_data: List[List[str]],
                            reference_data: List[List[str]],
                            source_dict: Dictionary,
                            reference_dict: Dictionary,
                            beam_size: int,
                            save_path: str = None):
    # get all model file paths in checkpoint path
    model_file_paths = glob.glob(f"{checkpoint_path}/*.pth")
    # sort for chronological order of checkpoints, if not already
    model_file_paths.sort()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    bleu_scores = []
    for model_path in model_file_paths:
        model = torch.load(model_path, map_location=device)
        # get bleu of model
        bleu_score = get_bleu_of_model(model,
                                       source_data,
                                       reference_data,
                                       source_dict,
                                       reference_dict,
                                       beam_size,
                                       do_beam_search=True,
                                       use_torch_bleu=True)

        bleu_scores.append(bleu_score)
        print(model_path, bleu_score)

    print(bleu_scores)

    if save_path is not None:
        file = open(save_path, "w")
        for bleu_score in bleu_scores:
            file.write(f" {bleu_score} \n")
        file.close()

    return bleu_scores


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('-cp', '--checkpoint_path', type=str)
    parser.add_argument('-sdp', '--source_data_path', type=str)
    parser.add_argument('-rdp', '--reference_data_path', type=str)
    parser.add_argument('-sd', '--source_dict_path', type=str)
    parser.add_argument('-rd', '--reference_dict_path', type=str)
    parser.add_argument('-bs', '--beam_size', type=int)
    parser.add_argument('-sp', '--save_path', type=str)

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_arguments()

    source_data = load_data(args.source_data_path)
    reference_data = load_data(args.reference_data_path)
    source_dict = Dictionary.load(args.source_dict_path)
    reference_dict = Dictionary.load(args.reference_dict_path)

    get_bleu_of_checkpoints(args.checkpoint_path,
                            source_data,
                            reference_data,
                            source_dict,
                            reference_dict,
                            args.beam_size,
                            args.save_path)
