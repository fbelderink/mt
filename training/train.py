import math

import torch
from torch.utils.data import DataLoader
from model.basic_net import BasicNet
from preprocessing.dataset import TranslationDataset


def train(train_path: str, validation_path: str, max_epochs=200, batch_size=200,
          shuffle=True, num_workers=4, lr=1e-3, eval_rate=100):
    """
    TODO
    - add tensorboard
    - do eval
    - add checkpoints for saving the model
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"training on {device}")

    train_set = TranslationDataset.load(train_path)
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    validation_set = TranslationDataset.load(validation_path)
    validation_dataloader = DataLoader(validation_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    model = BasicNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.print_structure()

    total_loss = 0
    total_steps = 0
    total_correct_predictions = 0

    for epoch in range(max_epochs):
        correct_predictions = 0
        steps = 0
        for S, T, L in train_dataloader:

            S = S.to(device)
            T = T.to(device)
            L = L.to(device)

            optimizer.zero_grad()

            pred = model(S, T)

            loss = model.compute_loss(pred, L)

            loss.backward()
            optimizer.step()

            # keep track of metrics
            steps += 1
            total_loss += loss.item()
            if torch.argmax(pred) == L:
                correct_predictions += 1

            accuracy = correct_predictions / steps
            perplexity = torch.exp(loss)

            # evaluate model every k updates
            if total_steps % eval_rate == 0:
                model.eval()
                for S_v, T_v, L_v in validation_dataloader:
                    S_v = S_v.to(device)
                    T_v = T_v.to(device)
                    L_v = L_v.to(device)

                    with torch.no_grad():
                        pred_v = model(S_v, T_v)
                        # do some eval

                model.train()

        # keep track of total metrics
        total_steps += steps
        total_correct_predictions += correct_predictions
        total_accuracy = (total_correct_predictions / total_steps)
        steps = 0
        correct_predictions = 0



