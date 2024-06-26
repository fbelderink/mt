import torch
from torch.utils.data import DataLoader
from model.basic_net import BasicNet
from preprocessing.dataset import TranslationDataset
from utils.hyperparameters import Hyperparameters
from multiprocessing import freeze_support

from utils.file_manipulation import save_checkpoint

# niedrigste perplexity: 12

def _count_correct_predictions(pred, L):
    correct_predictions = 0
    for p, l in zip(torch.argmax(pred, dim=1), L):
        if p == l:
            correct_predictions += 1
    return correct_predictions


def train(train_path: str, validation_path: str, config: Hyperparameters, max_epochs=100,
          shuffle=True, num_workers=4, val_rate=100, train_eval_rate=10):
    lr = config.learning_rate
    batch_size = config.batch_size

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"training on {device}")

    train_set: TranslationDataset = TranslationDataset.load(train_path)
    train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    print("Number of Batches: " + str(len(train_dataloader)))
    print("Batch Size: " + str(batch_size))

    #validation_set: TranslationDataset = TranslationDataset.load(validation_path)
    #validation_dataloader = DataLoader(validation_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    model = BasicNet(train_set.get_source_dict_size(), train_set.get_target_dict_size(), config,
                     window_size=train_set.get_window_size()).to(device)

    if config.saved_model != "":
        model = torch.load(config.saved_model)

    optimizer = config.optimizer(model.parameters(), lr=lr)

    model.print_structure()

    total_steps = 0
    previous_validation_perplexity = 0
    per_epoch = False
    per_batch = False
    checkpoints_rate = config.checkpoints

    if 0 < checkpoints_rate <= 1:
        # create checkpoints during epochs
        checkpoints_rate = 1 // checkpoints_rate
        per_epoch = True
        per_batch = False
    elif checkpoints_rate > 1:
        # create checkpoints during batch
        per_batch = True
        per_epoch = False

    if checkpoints_rate == 0:
        # no checkpoints
        per_batch = False
        per_epoch = False

    epoch_count = -(max_epochs % checkpoints_rate)

    print("\n        Starting Training: \n")
    for epoch in range(max_epochs):
        if per_epoch and epoch_count == checkpoints_rate:
            save_checkpoint(model)
            epoch_count = 0
        steps = 0

        batch_count = -(len(train_dataloader) % checkpoints_rate)

        for S, T, L in train_dataloader:
            if per_batch and batch_count == (len(train_dataloader) // checkpoints_rate):
                save_checkpoint(model)
                batch_count = 0

            S = S.to(device)
            T = T.to(device)
            L = L.long().to(device)

            optimizer.zero_grad()

            pred = model(S, T)

            loss = model.compute_loss(pred, L)

            loss.backward()
            optimizer.step()

            # keep track of metrics
            steps += 1
            total_steps += 1

            # print batch metrics
            batch_correct_predictions = _count_correct_predictions(pred, L)
            true_batch_size = L.size(0)
            batch_accuracy = batch_correct_predictions / true_batch_size
            batch_perplexity = torch.exp(loss)

            if per_batch:
                batch_count += 1

            if steps % train_eval_rate == 0:
                print("batch_accuracy:" + str(batch_accuracy))
                print("batch_perplexity:" + str(batch_perplexity.item()))
                print("epoch: " + str(epoch))
                print("steps: " + str(steps))
                print()

            # evaluate model every k updates
            '''if total_steps % val_rate == 0:
                model.eval()

                total_val_loss = 0
                total_val_correct_predictions = 0
                total_val_samples = 0

                # VALIDATION
                for S_v, T_v, L_v in validation_dataloader:
                    S_v = S_v.to(device)
                    T_v = T_v.to(device)
                    L_v = L_v.to(device)

                    total_val_samples += L_v.size(0)

                    pred_v = model(S_v, T_v)

                    total_val_correct_predictions += _count_correct_predictions(pred_v, L_v)

                    # compute cross entropy without averaging
                    total_val_loss += model.compute_loss(pred_v, L_v, False)

                validation_accuracy = total_val_correct_predictions / total_val_samples
                validation_perplexity = torch.exp(total_val_loss / total_val_samples).item()
                print()
                print("Validation:")
                print("Validation accuracy: " + str(validation_accuracy))
                print("Validation perplexity: " + str(validation_perplexity))

                if config.half_lr and 0 < previous_validation_perplexity <= validation_perplexity:
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = param_group['lr'] / 2
                    print("learning rate halfed; new learning rate: " + str(optimizer.param_groups[0]['lr']))

                previous_validation_perplexity = validation_perplexity
                print()

                model.train()'''

        if per_epoch:
            epoch_count += 1


if __name__ == '__main__':
    # call your train function here
    freeze_support()
    train("data/training_dataset_joint", "data/validation_dataset_joint")
