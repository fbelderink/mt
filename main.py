import argparse
from assignments.assignment1 import *
from assignments.assignment4 import *


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('-de', '--data_de_path', type=str)
    parser.add_argument('-en', '--data_en_path', type=str)
    parser.add_argument('-de_dev', '--data_de_dev_path', type=str)
    parser.add_argument('-en_dev', '--data_en_dev_path', type=str)
    parser.add_argument('-hy', '--hyps', type=str)
    parser.add_argument('-r', '--refs', type=str)
    parser.add_argument('-B', '--batch_size', type=int)
    parser.add_argument('-w', '--window_size', type=int)
    parser.add_argument('-bs', '--do_beam_search', type=bool)

    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_arguments()

    """
    hyps = load_data(args.hyps)
    refs = load_data(args.refs)

    first_assignment(args, hyps, refs)
    """

    data_de = load_data(args.data_de_path)
    data_en = load_data(args.data_en_path)

    data_de_dev = load_data(args.data_de_dev_path)
    data_en_dev = load_data(args.data_en_dev_path)

    dict_de = Dictionary.load("data/dicts/train_dict_de.pkl")
    dict_en = Dictionary.load("data/dicts/train_dict_en.pkl")

    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    #model = torch.load("eval/best_models/bleu28_6_w3.pth", map_location=device)

    #translations = test_beam_search(model, data_de_dev, dict_de, dict_en, 3, model.window_size, get_n_best=True)

    #translations = load_data("eval/translations/best_translations")

    #translations = test_greedy_search(model, data_de_dev, dict_de, dict_en, model.window_size)

    #test_get_scores(model, data_de_dev, data_en_dev, dict_de, dict_en, model.window_size)

    #test_model_bleu(model, data_de_dev, data_en_dev, dict_de, dict_en,
    #                3,  model.window_size, args.do_beam_search, translations, use_torch_bleu=True)

    #bleus = determine_models_bleu('eval/checkpoints/2024-06-12', data_de_dev, data_en_dev, dict_de, dict_en,
    #                              3, model.window_size, True)
    #print(f"Best model: {max(bleus)}")

    #our_score_avg, ref_score_avg = eval_scores(model, data_de_dev, data_en_dev, dict_de, dict_en,
    #                                           translations=translations)

    #print(our_score_avg, ref_score_avg)