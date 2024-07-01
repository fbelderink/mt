import argparse

import numpy as np

import torch
from torch.utils.data import DataLoader

from model.ff.feedforward_net import FeedforwardNet
from preprocessing.dataset.dataset import TranslationDataset
from utils.ConfigLoader import ConfigLoader
from utils.file_manipulation import save_checkpoint
from utils.model_hyperparameters import RNNModelHyperparameters, FFModelHyperparameters, \
    ModelHyperparameters
from utils.train_hyperparameters import TrainHyperparameters, RNNTrainHyperparameters, \
    FFTrainHyperparameters
from model.seq2seq.recurrent_net import RecurrentNet

from preprocessing.dictionary import PADDING


def train(train_path: str, validation_path: str,
          model_params: ModelHyperparameters, train_params: TrainHyperparameters,
          model_name=None):
    # train encoder and decoder separately using two optimizers (cf. Pytorch tutorial)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    train_set = TranslationDataset.load(train_path)
    train_dataloader = DataLoader(train_set,
                                  batch_size=train_params.batch_size,
                                  shuffle=train_params.shuffle,
                                  num_workers=train_params.num_workers)

    validation_set = TranslationDataset.load(validation_path)
    validation_dataloader = DataLoader(validation_set,
                                       batch_size=len(validation_set),
                                       shuffle=train_params.shuffle,
                                       num_workers=train_params.num_workers)

    optimizers = None
    if isinstance(model_params, RNNModelHyperparameters) and isinstance(train_params, RNNTrainHyperparameters):
        model = RecurrentNet(train_set.get_source_dict_size(),
                             train_set.get_target_dict_size(),
                             model_params,
                             model_name).to(device)

        if train_params.two_optimizers:
            encoder_optimizer = train_params.optimizer(model.get_encoder().parameters(), lr=train_params.learning_rate)
            decoder_optimizer = train_params.optimizer(model.get_decoder().parameters(), lr=train_params.learning_rate)

            optimizers = [encoder_optimizer, decoder_optimizer]
    elif isinstance(model_params, FFModelHyperparameters) and isinstance(train_params, FFTrainHyperparameters):
        model = FeedforwardNet(train_set.get_source_dict_size(),
                               train_set.get_target_dict_size(),
                               model_params,
                               window_size=train_set.get_window_size(),
                               model_name=model_name).to(device)
    else:
        raise ValueError('Invalid model hyperparameters')

    if train_params.saved_model != "":
        model = torch.load(train_params.saved_model)

    if not optimizers:
        optimizers = [train_params.optimizer(model.parameters(), lr=train_params.learning_rate)]

    print(f"training on {device}")
    print("Number of Batches: " + str(len(train_dataloader)))
    print("Batch Size: " + str(train_params.batch_size))
    model.print_structure()
    print("\nStarting Training:\n")

    total_steps = 0
    checkpoint_rate = train_params.checkpoints if train_params.checkpoints > 1 else 1 / train_params.checkpoints
    for epoch in range(1, train_params.max_epochs + 1):
        print(f"Epoch {epoch}/{train_params.max_epochs}")

        total_steps, epoch_loss = train_epoch(model,
                                              train_dataloader, validation_dataloader,
                                              optimizers,
                                              train_params, epoch, total_steps,
                                              checkpoint_rate, train_params.checkpoints > 1)

        if 0 < train_params.checkpoints <= 1 and epoch % checkpoint_rate == 0:
            save_checkpoint(model, model.model_name)

    save_checkpoint(model, model.model_name)


def train_epoch(model, train_dataloader, validation_dataloader,
                optimizers, train_params, epoch_num, total_steps,
                checkpoint_rate=None, do_checkpointing=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    steps = 0
    epoch_loss = 0
    previous_val_ppl = 0

    for source, target, label in train_dataloader:
        source = source.to(device)
        target = target.to(device)
        label = label.to(device)

        # TRAIN LOOP
        for optimizer in optimizers:
            optimizer.zero_grad()

        predictions, loss = forward_pass(model, source, target, label, train_params)

        loss.backward()

        for optimizer in optimizers:
            optimizer.step()

        epoch_loss += loss.item()

        # EVAL AND OPTIONAL FEATURES
        if steps % train_params.print_eval_every == 0:
            perplexity, accuracy = evaluate_performance(predictions,
                                                        loss.item(),
                                                        label)
            print(f"steps: {steps}, epoch: {epoch_num}")
            print(f"batch metrics: accuracy: {accuracy}, perplexity: {perplexity}, loss: {loss.item()}\n")
            #print(torch.argmax(predictions, dim=1)[1])
            #print(label[1])
        if train_params.test_model_every != 0 and steps % train_params.test_model_every == 0:
            val_ppl, val_acc = test_on_validation_data(model, validation_dataloader, train_params)

            #TODO use bleu
            if train_params.early_stopping and 0 < previous_val_ppl <= val_ppl:
                print("EARLY STOPPING")
                return

            if train_params.half_lr and 0 < previous_val_ppl <= val_ppl:
                half_lr(optimizers)
                print(
                    f"HALF LR: previous lr: {2 * optimizers[0].param_groups[0]['lr']}, new lr: {optimizers[0].param_groups[0]['lr']}\n")

            previous_val_ppl = val_ppl

        steps += 1
        total_steps += 1

        step_rate = len(train_dataloader) // checkpoint_rate
        if do_checkpointing and steps % step_rate == 0:
            save_checkpoint(model, model.model_name)

    return total_steps, epoch_loss


def test_on_validation_data(model, validation_dataloader, train_params):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model.eval()

    source, target, label = next(iter(validation_dataloader))
    # only one iteration as batch_size = len(validation_set)

    source = source.to(device)
    target = target.to(device)
    label = label.to(device)

    predictions, loss = forward_pass(model, source, target, label, train_params)

    validation_perplexity, validation_accuracy = evaluate_performance(predictions,
                                                                      loss.item(),
                                                                      label)

    print(
        f"validation results: accuracy: {validation_accuracy}, perplexity: {validation_perplexity}, loss: {loss.item()}\n")

    model.train()

    return validation_perplexity, validation_accuracy


def forward_pass(model, source, target, label, train_params):
    if isinstance(train_params, RNNTrainHyperparameters):
        predictions = model(source, target,
                            teacher_forcing=train_params.teacher_forcing)
    elif isinstance(train_params, FFTrainHyperparameters):
        predictions = model(source, target)
        predictions = predictions.unsqueeze(-1)
    else:
        raise ValueError('Invalid train hyperparameters')

    loss = model.compute_loss(predictions, label)

    return predictions, loss


def evaluate_performance(predictions, loss, labels):
    # predictions shape (B x dict_size x seq_len)
    # L shape (B x seq_len)
    correct = 0
    samples = 0
    for ps, ls in zip(torch.argmax(predictions, dim=1), labels):
        for p, l in zip(ps, ls):
            if l != PADDING:
                # only count matching non eos symbols
                if p == l:
                    correct += 1
                samples += 1

    return np.exp(loss), correct / samples


def half_lr(optimizers):
    for optimizer in optimizers:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] / 2


def _parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument('-tp', '--train_path', type=str)
    parser.add_argument('-vp', '--validation_path', type=str)
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('-params', '--train_params', type=str)
    parser.add_argument('-rnn', '--train_rnn', type=bool)

    return parser.parse_args()


if __name__ == '__main__':
    args = _parse_arguments()

    model_config = ConfigLoader(args.config).get_config()
    train_config = ConfigLoader(args.train_params).get_config()

    if args.train_rnn:
        train(args.train_path, args.validation_path,
              RNNModelHyperparameters(model_config), RNNTrainHyperparameters(train_config))
    else:
        train(args.train_path, args.validation_path,
              FFModelHyperparameters(model_config), FFTrainHyperparameters(train_config))
