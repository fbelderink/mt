import math

import torch
from torch.utils.data import DataLoader
from model.basic_net import BasicNet
from preprocessing.dataset import TranslationDataset
from utils.hyperparameters import Hyperparameters
from datetime import datetime
from pathlib import Path

def train(train_path: str, validation_path: str, config: Hyperparameters, max_epochs=200, batch_size=200,
          shuffle=False, num_workers=0, lr=1e-4, eval_rate=100, half_learningrate = True):
    """
    TODO
    - add tensorboard
    - do eval
    - add checkpoints for saving the model
    - half learning rate if perfomance stagenates on evaluation set
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"training on {device}")

    train_set = TranslationDataset.load(train_path)
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    print("number of Batches: " + str(len(train_dataloader)))

   # validation_set = TranslationDataset.load(validation_path)
   # validation_dataloader = DataLoader(validation_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    model =  BasicNet(batch_size,train_set.get_source_dict_size(),train_set.get_target_dict_size(),config).to(device)

    if(config.saved_model != ""):
        model.load_state_dict(torch.load(config.saved_model))
    optimizer = config.optimizer(model.parameters(), lr=lr)

    model.print_structure()

    total_loss = 0
    total_steps = 0
    total_correct_predictions = 0
    per_epoch = False
    per_batch = False
    checkpoints_rate = config.checkpoints


    if(checkpoints_rate <= 1 and checkpoints_rate > 0):
        checkpoints_rate = 1//checkpoints_rate
        per_epoch = True
        per_batch = False
    elif checkpoints_rate>1:
        per_batch = True
        per_epoch = False

    if checkpoints_rate == 0:
        per_batch = False
        per_epoch = False
    current_checkpoint = 0


    if(per_epoch):
        epoch_count = -(max_epochs % checkpoints_rate)

    for epoch in range(max_epochs):
        if(per_epoch and epoch_count == checkpoints_rate):
            date = datetime.today().strftime('%d-%m-%Y')
            time = datetime.today().strftime('%H_%M_%S')

            Path(f"training/checkpoints/{date}").mkdir(exist_ok=True)
            torch.save(model.state_dict(), f"training/checkpoints/{date}/{time}")
            epoch_count = 0
        steps = 0

        if(per_batch):
            batch_count = -(len(train_dataloader)%checkpoints_rate)
        for S, T, L in train_dataloader:
            if(per_batch and batch_count == (len(train_dataloader)//checkpoints_rate)):
                date = datetime.today().strftime('%d-%m-%Y')
                time = datetime.today().strftime('%H_%M_%S')
                Path(f"training/checkpoints/{date}").mkdir(exist_ok=True)
                torch.save(model.state_dict(), f"training/checkpoints/{date}/{time}")
                batch_count = 0


            S = S.to(device)
            T = T.to(device)

            L = L.long()

            L = L.to(device)

            optimizer.zero_grad()

            pred = model(S, T)

            loss = model.compute_loss(pred, L)

            loss.backward()
            optimizer.step()

            # keep track of metrics
            steps += 1
            total_loss += loss.item()
            batch_correct_predictions = 0
            for p, l in zip(torch.argmax(pred,dim=1),L):
                if p == l:
                    batch_correct_predictions += 1
            
            batch_accuracy = batch_correct_predictions / batch_size
            batch_perplexity = torch.exp(loss)

            print("batch_accuracy:" + str(batch_accuracy))
            print("batch_perplexity:" + str(batch_perplexity.item()))
            if per_batch:
                batch_count +=1

            # evaluate model every k updates
            """ 
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
            """



        # keep track of total metrics
        total_steps += steps
        total_correct_predictions += batch_correct_predictions
        if(per_epoch):
            epoch_count+=1
        total_accuracy = total_correct_predictions / (total_steps*batch_size)


