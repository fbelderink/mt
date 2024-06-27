import torch
import numpy as np
from model.seq2seq.recurrent_net import RecurrentNet
from model.ff.feedforward_net import FeedforwardNet
from preprocessing.dictionary import Dictionary, START_SYMBOL, END_SYMBOL, PADDING_SYMBOL
from preprocessing.batching.fragment import create_source_window_matrix
from typing import List
from torch import nn


def translate(model: nn.Module,
              source_data: List[List[str]],
              source_dict: Dictionary,
              target_dict: Dictionary,
              beam_size: int,
              window_size: int,
              get_n_best=False,
              alignment_factor=1):

    if isinstance(model, RecurrentNet):
        assert beam_size is not None, "Beam size must be specified for beam search"
        return translate_rnn(model,
                             source_data,
                             source_dict,
                             target_dict,
                             beam_size,
                             get_n_best)

    elif isinstance(model, FeedforwardNet):
        assert window_size is not None, "Window size must be specified for FF models"
        assert beam_size is not None, "Beam size must be specified for beam search"
        return translate_ff(model,
                            source_data,
                            source_dict,
                            target_dict,
                            beam_size,
                            window_size,
                            get_n_best,
                            alignment_factor)


def translate_rnn(model: RecurrentNet,
                  source_data: List[List[str]],
                  source_dict: Dictionary,
                  target_dict: Dictionary,
                  beam_size: int,
                  get_n_best=False):
    # initialization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = model.to(device)
    model.eval()

    # vectorization of dictionary functions
    get_source_index = np.vectorize(source_dict.get_index_of_string)
    get_target_index = np.vectorize(target_dict.get_index_of_string)
    get_target_string = np.vectorize(target_dict.get_string_at_index)

    source_data = source_dict.apply_vocabulary_to_text(source_data, bpe_performed=False)

    # search for longest sentence
    T_max = len(max(source_data, key=lambda s: len(s)))

    # padd each sentence to the length of the longest sentence, because we do that in training for parallel batch processing
    source_data = [sentence +
                   [END_SYMBOL] +
                   [PADDING_SYMBOL] * (T_max - len(sentence))
                   for sentence in source_data]

    # convert tokens to their respective indices
    source_data = torch.from_numpy(get_source_index(np.array(source_data)))

    target_sentences = []

    for sentence in source_data:
        # list to keep track the last token of each beam
        beam_targets = torch.from_numpy(get_target_index([[START_SYMBOL]])).to(device)

        # list to keep track of the overall probability of each beam
        top_k_values = [0] * beam_size
        #
        top_k_indices = [[target_dict.get_index_of_string(START_SYMBOL)]] * beam_size

        # roll out encoder
        encoder_outputs, state = model.get_encoder().forward(sentence)

        # TODO: once decoder is unidirectional, both encoder states have to be concatenated
        # add batch dimensions
        state = (state[0].unsqueeze(1), state[1].unsqueeze(1))

        # add batch dimension
        encoder_outputs = encoder_outputs.unsqueeze(0)

        # list for states of each beam
        states = [state] * beam_size

        for k in range(T_max + 1):

            # list to temporarily store the top k choices of each beam
            all_top_k_indices = []
            all_top_k_values = []
            new_states = []

            # TODO eos beachten und in extra liste auslagern, danach topk über anderen top k (also top k ohne eos)
            for beam_idx, target in enumerate(beam_targets):
                # add batch dimension
                target = target.unsqueeze(0)

                # do one step on the last token of the beam
                pred, state = model.get_decoder().forward_step(encoder_outputs,
                                                               states[beam_idx],
                                                               target)
                new_states.append(state)
                # add previous top k values along beam size dim
                pred += top_k_values[beam_idx]
                # längennormalisierung der predicitions
                pred /= k + 1

                # get top k predictions
                top_k = pred.topk(beam_size, dim=-1)

                all_top_k_indices.extend(top_k.indices.flatten())
                all_top_k_values.extend(top_k.values.flatten())

            new_indices = torch.stack(all_top_k_indices).topk(beam_size, dim=-1)

            # saving the top k values in a list
            top_k_values = [all_top_k_values[i].item() for i in new_indices.indices.flatten().tolist()]
            current_top_k_indices = [idx for idx in new_indices.values.flatten().tolist()]

            # proposed fix:
            beam_indices = [idx // beam_size for idx in new_indices.indices.flatten().tolist()]

            top_k_indices = [top_k_indices[beam_index] + [current_top_k_indices[beam_index]]
                             for beam_index in beam_indices]

            beam_targets = torch.from_numpy(np.array([indices[-1] for indices in top_k_indices])).to(device).unsqueeze(1)
            if k == 0:
                states = new_states * beam_size
            else:
                states = [new_states[i] for i in beam_indices]

            # get target translation (first entries are sos)
        if not get_n_best:
            target_sentence = get_target_string(top_k_indices[np.argmax(top_k_values)][1:]).tolist()
            target_sentences.append(target_sentence)
        else:
            target_sentences = get_target_string([indices[1:] for indices in top_k_indices]).tolist()  # remove sos

            target_sentences.append(target_sentences)

    return target_sentences


def translate_ff(model: FeedforwardNet,
                 source_data: List[List[str]],
                 source_dict: Dictionary,
                 target_dict: Dictionary,
                 beam_size: int,
                 window_size: int,
                 get_n_best=False,
                 alignment_factor=1):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.eval()

    source_data = source_dict.apply_vocabulary_to_text(source_data, bpe_performed=False)

    get_target_idx = np.vectorize(target_dict.get_index_of_string)
    get_target_string = np.vectorize(target_dict.get_string_at_index)

    target_sentences = []

    for sentence in source_data:
        S = create_source_window_matrix(sentence, source_dict, window_size, len(sentence) * alignment_factor + 1)

        S = torch.from_numpy(S).to(device)

        beam_targets = torch.from_numpy(get_target_idx([[START_SYMBOL] * window_size])).to(device)

        top_k_values = [0]
        top_k_indices = [[target_dict.get_index_of_string(START_SYMBOL)] * window_size] * beam_size

        for s in S:
            # expand to make first dimension fit beam_size
            s = s.expand((beam_targets.shape[0], *s.shape))

            # predict on all previous top k results simultaneously
            pred = model(s, beam_targets)

            # add previous top k values along beam size dim
            pred += torch.tensor(top_k_values).unsqueeze(1).to(device)

            # flatten predictions to get top k along all previous top k predictions
            # (1, beam_size * vocab_size)
            pred = pred.reshape((1, beam_targets.shape[0] * pred.shape[-1]))

            # get top k predictions
            top_k = pred.topk(beam_size, dim=-1)

            if beam_targets.shape[0] == 1:
                # only applies in first iteration, when beam targets have one dim only
                beam_targets = beam_targets.repeat(beam_size, 1)

            # we flattened the prediction vector, so find out to which previous predictions new top k indices belong
            previous_indices = [i // len(target_dict) for i in top_k.indices.squeeze(0).tolist()]

            # get indices in target vocab range (also has to happen because of flattening)
            current_indices = [i % len(target_dict) for i in top_k.indices.squeeze(0).tolist()]

            # init list to record new top k predictions
            new_top_k_indices = []

            for i in range(beam_size):
                # select correct previous list and add new index
                new_top_k_indices.append(top_k_indices[previous_indices[i]] + [current_indices[i]])

                # replace indices in input for net
                beam_targets[i] = torch.tensor(new_top_k_indices[i][-window_size:]).to(device)

            # record current best values
            top_k_values = top_k.values.squeeze(0).tolist()
            # replace top k indices with newly found best indices
            top_k_indices = new_top_k_indices

        # get target translation (first window_size entries are sos)
        if not get_n_best:
            target_sentences.append(get_target_string(top_k_indices[np.argmax(top_k_values)][window_size:]).tolist())
        else:
            target_sentences.append(get_target_string([indices[window_size:] for indices in top_k_indices]).tolist())

    return target_sentences
