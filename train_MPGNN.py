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
from config import DATASET_PATH, PATIENCE, SEED, GRID_COMBINATIONS, get_result_path
torch.manual_seed(SEED)
random.seed(SEED)


class PrefixIGs(InMemoryDataset):
    def __init__(self, ds_path):
        super(PrefixIGs, self).__init__()
        self.data, self.slices = torch.load(ds_path)


if __name__ == '__main__':
    model_type = 'MPGNN'
    datasets = [join(DATASET_PATH, name) for name in listdir(DATASET_PATH) if name.endswith('.pt')]
    for dataset_path in datasets:
        print(f'Processing dataset: {dataset_path}')
        ds_name = dataset_path.split('/')[-1].split('.')[0]
        ds_activities = f'{dataset_path[:-3]}_activities.txt'
        result_path = get_result_path(ds_name, model_type)
        run_time = datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
        run_path = join(result_path, run_time)
        G = PrefixIGs(dataset_path)
        train_dataset = [data for data in G if data.set == 'train']
        test_dataset = [data for data in G if data.set == 'test']
        actual_comb, total_combs = 0, len(GRID_COMBINATIONS)
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

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            print(f'Running on device: {device}')

            train_dataset = [data.to(device) for data in train_dataset]
            train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size)

            test_dataset = [data.to(device) for data in test_dataset]
            test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size)


            model = MPGNN(dataset=G, num_layers=graph_conv_layers, dropout=dropout, num_neurons=num_neurons, k=k)
            model = model.to(device)

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
                for batch in train_loader:
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

                model.eval()
                with torch.no_grad():
                    for batch in test_loader:
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

                if no_improvements > PATIENCE:
                    print(f"** EARLY STOPPING AT EPOCH: {epoch} **")
                    break

            calculate_combination_metrics(results_df, comb_path)
            print('Combination done!\n')
            actual_comb += 1
        print('All done!')
        print('\n\nCalculating best results...')
        best_combinations(model_type, ds_name)
