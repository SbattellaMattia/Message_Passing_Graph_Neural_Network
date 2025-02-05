import os

import pandas as pd
from time import time
import torch
import hashlib
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
import random
from datetime import datetime
from MPGNN import MPGNN
import numpy as np
from os import listdir, makedirs
from os.path import join
from results_evaluation import (best_combinations, calculate_metrics_by_prefixes, calculate_combination_metrics,
                                plot_confusion_matrix)
from config import DATASET_PATH, PATIENCE, SEED, GRID_COMBINATIONS, get_result_path, THRESHOLD, NO_REPEAT
torch.manual_seed(SEED)
random.seed(SEED)


class PrefixIGs(InMemoryDataset):
    def __init__(self, ds_path):
        super(PrefixIGs, self).__init__()
        self.data, self.slices = torch.load(ds_path)


if __name__ == '__main__':
    model_type = 'MPGNN'
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f'Running on device: {device}')
    datasets = [join(DATASET_PATH, name) for name in listdir(DATASET_PATH) if name.endswith('.pt')]
    for dataset_path in datasets:
        #if dataset_path==datasets[0]: continue
        #print(f'Processing dataset: {dataset_path}')
        #ds_name = dataset_path.split('/')[-1].split('.')[0]
        ds_name = dataset_path.split('\\')[-1].split('.')[0]
        print(f'Processing dataset: {ds_name}')
        ds_activities = f'{dataset_path[:-3]}_activities.txt'
        result_path = get_result_path(ds_name, model_type)
        run_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
        run_path = join(result_path, run_time)
        G = PrefixIGs(dataset_path)
        train_dataset = [data for data in G if data.set == 'train']
        #train_dataset = [data.to(device) for data in train_dataset]
        test_dataset = [data for data in G if data.set == 'test']
        #test_dataset = [data.to(device) for data in test_dataset]
        actual_comb, total_combs = 0, len(GRID_COMBINATIONS)

        #Aggiunta del file con le combinazioni effettuate
        completed_file = "completed_combinations.txt"
        skipped_combinations = 0
        if not os.path.exists(completed_file):
            with open(completed_file, 'w') as f:
                pass
        else:
            with open(completed_file, 'r') as f:
                completed_combinations = set(line.strip() for line in f.readlines())

        for comb in GRID_COMBINATIONS:
            batch_size = comb['batch_size']
            epochs = comb['epochs']
            k = comb['k']
            num_neurons = comb['num_neurons']
            graph_conv_layers = comb['graph_conv_layers']
            learning_rate = comb['learning_rate']
            dropout = comb['dropout']
            comb_string = comb['path']
            comb_path = join(run_path, comb_string)
            makedirs(comb_path, exist_ok=True)

            already_in = False
            #Se la combinazione è stata effettuata salta alla prossima
            if (ds_name + "_" + comb_string) in completed_combinations:
                already_in = True
                if NO_REPEAT:
                    print(f"Skipping combination {ds_name}_{comb_string} (already completed)")
                    skipped_combinations += 1
                    actual_comb += 1
                    continue

            #train_dataset = [data.to(device) for data in train_dataset]
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, num_workers=0)

            #test_dataset = [data.to(device) for data in test_dataset]
            test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, num_workers=0)

            model = MPGNN(dataset=G, num_layers=graph_conv_layers, dropout=dropout, num_neurons=num_neurons, k=k)
            #model=model.to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
            criterion = torch.nn.CrossEntropyLoss()

            results_df = pd.DataFrame(columns=['run', 'combination', 'epoch', 'train_acc', 'test_acc', 'train_f1',
                                               'test_f1', 'train_loss', 'test_loss', 'best_test_loss'])
            print(f'\n** NEW COMBINATION STARTED ({actual_comb + 1}/{total_combs}) **')
            no_improvements, best_test_loss = 0, np.inf
            for epoch in range(epochs):
                epoch_path = join(comb_path, f'{epoch}')
                makedirs(epoch_path, exist_ok=True)
                train_prefix_results = pd.DataFrame(columns=['epoch', 'prefix_len', 'y_estimated', 'y_real'])
                test_prefix_results = pd.DataFrame(columns=['epoch', 'prefix_len', 'y_estimated', 'y_real'])
                epoch_prefix_results = pd.DataFrame()
                start_epoch_time = time()

                loss_train, loss_test = 0, 0
                model.train()
                #print("Start training...")
                for batch in train_loader:
                    batch = batch.to(device)
                    pred = model(batch)
                    class_pred, class_real = pred.argmax(dim=1), batch.y.argmax(dim=1)

                    # Loss
                    loss = criterion(pred, class_real)
                    loss_train += loss.item()

                    # Backpropagation
                    loss.backward()  # compute parameters gradients
                    optimizer.step()  # update parameters
                    optimizer.zero_grad()  # reset the gradients of all parameters

                    # Epoch metrics
                    train_prefix_results = pd.concat([train_prefix_results, pd.DataFrame(
                        {'epoch': [epoch] * len(batch.prefix_len),
                         'prefix_len': batch.prefix_len.cpu().numpy(),
                         'y_estimated': class_pred.cpu().numpy(),
                         'y_real': class_real.cpu().numpy()
                         })], ignore_index=True)

                loss_train /= len(train_loader.dataset)
                train_metrics = calculate_metrics_by_prefixes(train_prefix_results, epoch_path, 'train')
                plot_confusion_matrix(train_prefix_results, ds_activities, epoch_path, 'train')


                #print("evalutation model")
                model.eval()
                #print("Start testing...")
                with torch.no_grad():
                    for batch in test_loader:
                        batch = batch.to(device)
                        pred = model(batch)
                        class_pred, class_real = pred.argmax(dim=1), batch.y.argmax(dim=1)

                        # Loss
                        loss = criterion(pred, class_real)
                        loss_test += loss.item()

                        # Epoch metrics
                        test_prefix_results = pd.concat([test_prefix_results, pd.DataFrame(
                            {'epoch': [epoch] * len(batch.prefix_len),
                             'prefix_len': batch.prefix_len.cpu().numpy(),
                             'y_estimated': class_pred.cpu().numpy(),
                             'y_real': class_real.cpu().numpy()
                             })], ignore_index=True)

                loss_test /= len(test_loader.dataset)
                epoch_time = time() - start_epoch_time
                test_metrics = calculate_metrics_by_prefixes(test_prefix_results, epoch_path, 'test')
                plot_confusion_matrix(test_prefix_results, ds_activities, epoch_path, 'test')

                # salviamo le predizioni dei prefissi per l'epoca corrente
                train_prefix_results['set'] = 'train'
                test_prefix_results['set'] = 'test'
                epoch_prefix_results = pd.concat([train_prefix_results, test_prefix_results])
                epoch_prefix_results.to_csv(join(epoch_path, f'{epoch}_prefix_results.csv'), index=False)

                # salviamo i risultati dell'epoca corrente
                results_df.loc[len(results_df)] = [run_time, comb_string, epoch, train_metrics['acc'],
                                                   test_metrics['acc'],
                                                   train_metrics['f1'], test_metrics['f1'], loss_train, loss_test,
                                                   str(loss_test < best_test_loss)]

                #print("stamp results in csv...")

                total_path=join(comb_path, f'results_{hashlib.md5(comb_string.encode()).hexdigest()[:8]}.csv') 
                results_df.to_csv(total_path, header=True, sep=',', index=False)
                #results_df.to_csv(join(comb_path, f'results_{comb_string}.csv'), header=True, sep=',', index=False)

                if loss_test < best_test_loss:
                    no_improvements = 0
                    best_test_loss = loss_test
                    print(f' ** BEST TEST LOSS {round(best_test_loss, 4)} **')
                else:
                    no_improvements += 1

                torch.save(model.state_dict(), join(epoch_path, f'{epoch}_model.pt'))
                print(f"Epoch: {epoch}/{epochs} | Time: {round(epoch_time, 2)}s "
                      f"| Train loss: {loss_train:.4f} | Test loss: {loss_test:.4f}\nTRAIN: {train_metrics}\nTEST: {test_metrics}\n")

                #soppressione se iperparametri non promettenti
                if test_metrics.get('acc') < THRESHOLD or test_metrics.get('f1') < THRESHOLD: break

                if no_improvements > PATIENCE:
                    print(f"** EARLY STOPPING AT EPOCH: {epoch} **")
                    break

            if not already_in:
                with open(completed_file, 'a') as f:
                    f.write(f"{ds_name}_{comb_string}\n")
                print('Saving...')

            calculate_combination_metrics(results_df, comb_path)
            print('Combination done!\n')
            actual_comb += 1
        print('All done!')

        # Registra la combinazione completata
        # NOTA: indipendentemente dal fatto che NO_REPEAT sia attivo
        #       o meno la salva, a meno che non sia già presente

        if skipped_combinations!=len(GRID_COMBINATIONS):
            print('\n\nCalculating best results...')
            best_combinations(model_type, ds_name)
