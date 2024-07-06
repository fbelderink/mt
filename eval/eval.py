import argparse
import glob
from utils.file_manipulation import *
from metrics.calculate_bleu_of_model import get_bleu_of_model
from search import beam_search
from scoring import score
from preprocessing.dictionary import Dictionary
from model.basic_net import BasicNet
from model.seq2seq.recurrent_net import RecurrentNet
from model.ff.feedforward_net import FeedforwardNet
from postprocessing.postprocessing import undo_prepocessing
from search.beam_search import translate_rnn, translate_ff
from torchtext.data.metrics import bleu_score as torch_bleu


def translate(model_path: str,
              source_data: List[List[str]],
              source_dict: Dictionary,
              target_dict: Dictionary,
              beam_size: int,
              get_n_best=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = torch.load(model_path, map_location=device)

    if isinstance(model, RecurrentNet):
        target_sentences = translate_rnn(model,
                                         source_data,
                                         source_dict,
                                         target_dict,
                                         beam_size,
                                         get_n_best=get_n_best)
    elif isinstance(model, FeedforwardNet):
        target_sentences = translate_ff(model,
                                        source_data,
                                        source_dict,
                                        target_dict,
                                        beam_size,
                                        window_size=model.window_size,
                                        get_n_best=get_n_best)
    else:
        raise ValueError("unsupported model type")

    if get_n_best:
        post_processed_sentences = []
        for sentences in target_sentences:
            post_processed_sentences.append(undo_prepocessing(sentences))
    else:
        post_processed_sentences = undo_prepocessing(target_sentences)

    if get_n_best:
        save_n_best_translations(f"eval/translations/beam_translations_n_best_{model.model_name}",
                                 post_processed_sentences)
    else:
        save_data(f"eval/translations/beam_translations_{model.model_name}", post_processed_sentences)

    return post_processed_sentences


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

    if save_path is not None:
        file = open(save_path, "w")
        for bleu_score in bleu_scores:
            file.write(f" {bleu_score} \n")
        file.close()

    return bleu_scores


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('-mp', '--model_path', type=str)
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

    if args.model_path:
        translations = translate(args.model_path,
                                 source_data,
                                 source_dict,
                                 reference_dict,
                                 beam_size=args.beam_size,
                                 get_n_best=False)

        print(torch_bleu(translations, [[ref] for ref in reference_data]))

    elif args.checkpoint_path:
        get_bleu_of_checkpoints(args.checkpoint_path,
                                source_data,
                                reference_data,
                                source_dict,
                                reference_dict,
                                args.beam_size,
                                args.save_path)
