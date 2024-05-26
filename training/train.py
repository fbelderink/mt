import math

import torch
from torch.utils.data import DataLoader
from model.basic_net import BasicNet
from preprocessing.dataset import TranslationDataset
from multiprocessing import freeze_support

def _count_correct_predictions(pred, L):
    correct_predictions = 0
    for p, l in zip(torch.argmax(pred,dim=1),L):
        if p == l:
           correct_predictions += 1
    return correct_predictions
            


def train(train_path: str, validation_path: str, max_epochs=400, batch_size=200,
          shuffle=True, num_workers=0, lr=1e-3, eval_rate=100, half_learningrate = True):
    """
    TODO
    - add tensorboard
    - add checkpoints for saving the model
    - optinally half learning rate if perfomance stagenates on evaluation set
    """

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"training on {device}")

    train_set = TranslationDataset.load(train_path)
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    print("number of Batches: " + str(len(train_dataloader)))

    validation_set = TranslationDataset.load(validation_path)
    validation_dataloader = DataLoader(validation_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    model = BasicNet(batch_size,train_set.get_source_dict_size(),train_set.get_target_dict_size()).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    model.print_structure()

    total_loss = 0
    total_steps = 0
    total_correct_predictions = 0

    previous_validation_perplexity = 0

    for epoch in range(max_epochs):
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
            total_steps += 1
            total_loss += loss.item()


            # print batch metrics
            batch_correct_predictions = _count_correct_predictions(pred, L)
            true_batch_size = L.size(0)
            batch_accuracy = batch_correct_predictions / true_batch_size
            batch_perplexity = torch.exp(loss)

            if steps % 10 == 0:
                print("batch_accuracy:" + str(batch_accuracy))
                print("batch_perplexity:" + str(batch_perplexity.item()))
                print("epoch: " + str(epoch))
                print("steps: " + str(steps))
            
            # evaluate model every eval_rate updates
            if total_steps % eval_rate == 0:
                model.eval()

                #total_validation_perplexity = 0
                summed_cross_entropy = 0
                total_validation_correct_predictions = 0
                total_number_of_validation_samples = 0

                for S_v, T_v, L_v in validation_dataloader:
                    S_v = S_v.to(device)
                    T_v = T_v.to(device)
                    L_v = L_v.to(device)

                    total_number_of_validation_samples += L_v.size(0)
                    with torch.no_grad():
                        pred_v = model(S_v, T_v)

                        total_validation_correct_predictions += _count_correct_predictions(pred_v, L_v)

                        # compute cross entropy without averaging 
                        summed_cross_entropy += model.compute_loss(pred_v, L_v, False)

                
                                
                validation_accuracy = total_validation_correct_predictions / total_number_of_validation_samples
                validation_perplexity = torch.exp(summed_cross_entropy / total_number_of_validation_samples).item()    
                print()
                print("Validation:")
                print("Validation accuracy: " + str(validation_accuracy))
                print("Validation perplexity: " + str(validation_perplexity))
                

                if previous_validation_perplexity != 0 and previous_validation_perplexity <= validation_perplexity:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] / 2
                    print("learning rate halfed; new learning rate: "+ str(optimizer.param_groups[0]['lr']))

                previous_validation_perplexity = validation_perplexity
                print()
                model.train()

                
            



        # keep track of total metrics
        total_correct_predictions += batch_correct_predictions
        total_accuracy = total_correct_predictions / (total_steps*batch_size)



if __name__ == '__main__':
    # call your train function here
    freeze_support()
    train("data/training_dataset_joint", "data/validation_dataset_joint")
    