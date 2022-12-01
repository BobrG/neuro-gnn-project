import os
import sys
import random
import argparse
import numpy as np
from typing import Tuple, Optional, List

import nni
from torch import Tensor
from sklearn import metrics
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from torch_geometric.data import DataLoader


from models.build_model import build_model
from explainer import GNNExplainer
from utils.load_node_labels import load_txt, load_cluster_info_from_txt
from utils.utils import seed_everything, archive_files, mkdirs_if_needed
from utils.dataloader_utils import *
from utils.modified_args import ModifiedArgs
from utils.diff_matrix import DiffMatrix

from comet_ml import Experiment

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MainExplainer:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.experiment = None

    def train_and_evaluate(self, model, train_loader, test_loader, optimizer, device, args, is_tuning):
        model.train()
        accs, aucs, macros = [], [], []
        epoch_num = self.get_epoch_num(args, is_tuning)

        for i in range(epoch_num):
            loss_all = 0
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                out = model(data)
                loss = F.nll_loss(out, data.y)

                loss.backward()
                optimizer.step()

                loss_all += loss.item()
            epoch_loss = loss_all / len(train_loader.dataset)

            train_micro, train_auc, train_macro = self.eval(model, train_loader)
            
            title = "Tuning Train" if is_tuning else "Initial Train"
            print(f'({title}) | Epoch={i:03d}, loss={epoch_loss:.4f}, \n'
                  f'train_micro={(train_micro * 100):.2f}, train_macro={(train_macro * 100):.2f}, '
                  f'train_auc={(train_auc * 100):.2f}')
            if self.experiment is not None:
                self.experiment.log_metric(f'{title} AUC', (train_auc * 100), step=i)
                self.experiment.log_metric(f'{title} Loss', (epoch_loss), step=i)
            if (i + 1) % args.test_interval == 0 and test_loader is not None:
                test_micro, test_auc, test_macro = self.eval(model, test_loader)
                accs.append(test_micro)
                aucs.append(test_auc)
                macros.append(test_macro)
                text = f'({title} Epoch {i}), test_micro={(test_micro * 100):.2f}, ' \
                       f'test_macro={(test_macro * 100):.2f}, test_auc={(test_auc * 100):.2f}\n'
                print(text)
                with open(args.save_result, "a") as f:
                    f.writelines(text)
                if self.experiment is not None:
                    self.experiment.log_metric(f'{title.split(" ")[0]} Test AUC', (test_auc * 100), step=i)

        if args.enable_nni:
            nni.report_intermediate_result(accs[-1]) # report on test score after last training epoch

        # sort <==> authors take results from the best epoch
        accs, aucs, macros = numpy.sort(numpy.array(accs)), numpy.sort(numpy.array(aucs)), \
                             numpy.sort(numpy.array(macros))

        return accs.mean(), aucs.mean(), macros.mean()

    def get_epoch_num(self, args, is_tuning):
        if is_tuning:
            epoch_num = args.tuning_epochs
        else:
            epoch_num = args.initial_epochs
        return epoch_num

    @torch.no_grad()
    def eval(self, model, loader, test_loader: Optional[DataLoader] = None): # -> (float, float):
        model.eval()
        preds, trues, preds_prob = [], [], []

        for data in loader:
            data = data.to(self.device)
            c = model(data)

            pred = c.max(dim=1)[1]
            preds += pred.detach().cpu().tolist()
            preds_prob += torch.exp(c)[:, 1].detach().cpu().tolist()
            trues += data.y.detach().cpu().tolist()

        fpr, tpr, _ = metrics.roc_curve(trues, preds_prob)
        train_auc = metrics.auc(fpr, tpr)
        if numpy.isnan(train_auc):
            print('NaNs in train auc')
            train_auc = 0.5
        train_micro = f1_score(trues, preds, average='micro')
        train_macro = f1_score(trues, preds, average='macro', labels=[0, 1])

        if test_loader is not None:
            test_micro, test_auc, test_macro = self.eval(model, test_loader)
            return train_micro, train_auc, train_macro, test_micro, test_auc, test_macro
        else:
            return train_micro, train_auc, train_macro

    def explain(self, model, train_set, test_set, node_labels, node_atts, optimizer, device, \
                args, return_explainer=False, return_loaders=False):
        # the dataloader to train/test the explainer mask must be of batch size 1
        train_iterator = DataLoader(train_set, batch_size=1, shuffle=False)
        test_iterator = DataLoader(test_set, batch_size=1, shuffle=False)

        # train explainer mask
        explainer = GNNExplainer(model, epochs=args.explainer_epochs, return_type='log_prob', labels=node_labels,
                                 remove_loss=[args.remove_loss])
        node_feat_mask, edge_mask = explainer.explainer_train(train_iterator, device, args)
        # Tuning: used explainer masked loader to train the initial model again
        masked_train_loader = explainer.mask_dataloader(node_feat_mask, edge_mask, train_iterator, args, node_atts,
                                                        device, batch_size=args.train_batch_size)
        masked_test_loader = explainer.mask_dataloader(node_feat_mask, edge_mask, test_iterator, args, node_atts,
                                                       device, batch_size=args.test_batch_size) if test_set is not None else None

        self.train_and_evaluate(explainer.model, masked_train_loader, masked_test_loader, optimizer,
                                device, args, is_tuning=True)
        if test_set is not None:
            explainer_test_micro, explainer_test_auc, explainer_test_macro = self.eval(explainer.model,
                                                                                       masked_test_loader)

            print(f'(Tuning Performance Last Epoch) | explainer_test_micro={(explainer_test_micro * 100):.2f}, '
                  f'explainer_test_macro={(explainer_test_macro * 100):.2f}, '
                  f'explainer_test_auc={(explainer_test_auc * 100):.2f}')
#             experiment.log_metric('Tunning Test AUC', (explainer_test_auc * 100), step=args.tuning_epochs)
            
        if not return_explainer:
            return explainer_test_micro, explainer_test_auc, explainer_test_macro
        
        ret = [explainer, node_feat_mask, edge_mask]
        if return_loaders:
            ret += [masked_train_loader, masked_test_loader]
        return ret

    def dropout_samples(self, train_set: List[Data], train_y: Tensor, dropout_rate) -> Tuple[List[Data], Tensor]:
        # randomly shuffle the list of data and train_y together
        indices = torch.randperm(len(train_set))
        train_set = [train_set[i] for i in indices]
        train_y = train_y[indices]

        # dropout the data according to the dropout rate
        dropout_num = int(dropout_rate * len(train_set))
        train_set = train_set[dropout_num:]
        train_y = train_y[dropout_num:]

        return train_set, train_y
    
    def run_cv(self, args, dataset_list, y, node_atts, node_labels):
        dataset = MaskableList(dataset_list)
        num_features = dataset[0].x.shape[1]
        accs, aucs, macros = [], [], []
        exp_accs, exp_aucs, exp_macros = [], [], []
        for _ in range(args.repeat):
            seed_everything(args.seed)  # use random seed for each run
            skf = StratifiedKFold(n_splits=args.k_fold_splits, shuffle=True)
            for train_index, test_index in skf.split(dataset, y):
                model = build_model(args, device, num_features)
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                # authors datasplitting :/
                train_binary, test_binary = numpy.zeros(len(dataset), dtype=int), numpy.zeros(len(dataset), dtype=int)
                train_binary[train_index] = 1
                test_binary[test_index] = 1
                train_set: MaskableList[Data]
                train_set, test_set = dataset[train_binary], dataset[test_binary]
                train_y, test_y = y[train_index], y[test_index]

                if args.dropout_rate > 0:
                    train_set, train_y = self.dropout_samples(train_set, train_y, args.dropout_rate)

                if args.shallow == 'diff_matrix':
                    diff_matrix = DiffMatrix(0.2).compute(train_set, y)
                    _, train_set = diff_matrix.apply(train_set, args, train_y)
                    _, test_set = diff_matrix.apply(test_set, args, test_y)

                train_loader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=False)
                test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False)

                # train
                test_micro, test_auc, test_macro = self.train_and_evaluate(model, train_loader, test_loader,
                                                                           optimizer, device, args, is_tuning=False)

                print(f'(Initial Performance Last Epoch) | test_micro={(test_micro * 100):.2f}, '
                      f'test_macro={(test_macro * 100):.2f}, test_auc={(test_auc * 100):.2f}')

                accs.append(test_micro)
                aucs.append(test_auc)
                macros.append(test_macro)

                if args.explain:
                    explainer_test_micro, explainer_test_auc, explainer_test_macro = \
                        self.explain(model, train_set, test_set, node_labels, node_atts, optimizer, device, args)
                    exp_accs.append(explainer_test_micro)
                    exp_aucs.append(explainer_test_auc)
                    exp_macros.append(explainer_test_macro)

        result_str = f'(K Fold Initial)| avg_acc={(numpy.mean(accs) * 100):.2f} +- {(numpy.std(accs) * 100): .2f}, ' \
                     f'avg_auc={(numpy.mean(aucs) * 100):.2f} +- {numpy.std(aucs) * 100:.2f}, ' \
                     f'avg_macro={(numpy.mean(macros) * 100):.2f} +- {numpy.std(macros) * 100:.2f}\n'
        print(result_str)
        if args.explain:
            result_str += f'(K Fold Tuning) | avg_acc={(numpy.mean(exp_accs) * 100):.2f} +- {(numpy.std(exp_accs) * 100): .2f}, ' \
                          f'avg_auc={(numpy.mean(exp_aucs) * 100):.2f} +- {numpy.std(exp_aucs) * 100:.2f}' \
                          f'avg_macro={(numpy.mean(exp_macros) * 100):.2f} +- {numpy.std(exp_macros) * 100:.2f}'
            print(result_str)

        #with open('result.log', 'a') as f:
        #    # write all input arguments to f
        #    input_arguments: List[str] = sys.argv
        #    f.write(f'{input_arguments}\n')
        #    f.write(result_str + '\n')

        if args.enable_nni:
            nni.report_final_result(numpy.mean(aucs))
    
    def main(self):
        mkdirs_if_needed(["fig/", "fig/archive/", "modularity/", "modularity/archive/"])
        archive_files("fig/", "fig/archive/")
        archive_files("modularity/", "modularity/archive/")

        parser = argparse.ArgumentParser()
        parser.add_argument('--hidden_dim', type=int, default=16)
        parser.add_argument('--n_GNN_layers', type=int, default=2)
        parser.add_argument('--n_MLP_layers', type=int, default=1)
        parser.add_argument('--lr', type=float, default=0.001)
#         parser.add_argument('--num_heads', type=int, default=1)
        parser.add_argument('--weight_decay', type=float, default=1e-5)
        parser.add_argument('--initial_epochs', type=int, default=100)
        parser.add_argument('--explainer_epochs', type=int, default=100)
        parser.add_argument('--tuning_epochs', type=int, default=100)
        parser.add_argument('--test_interval', type=int, default=20)
        parser.add_argument('--seed', type=int, default=112078)
        parser.add_argument('--save_result', type=str, default='test')
        parser.add_argument('--node_features', type=str,
                            choices=['identity', 'LDP', 'node2vec', 'adj', 'diff_matrix'],
                            # LDP is degree profile and adj is edge profile
                            default='adj')
        parser.add_argument('--pooling', type=str,
                            choices=['sum', 'concat', 'mean'],
                            default='sum')
        parser.add_argument('--explain', action='store_true')
        parser.add_argument('--modality', type=str, default='dti')
        parser.add_argument('--dataset_name', type=str, default="BP")
        parser.add_argument('--train_batch_size', type=int, default=16)
        parser.add_argument('--test_batch_size', type=int, default=16)
        parser.add_argument('--k_fold_splits', type=int, default=7)
#         parser.add_argument('--gat_hidden_dim', type=int, default=8)
        parser.add_argument('--enable_nni', action='store_true')
        parser.add_argument('--dropout', type=float, default=0.5)
        parser.add_argument('--edge_emb_dim', type=int, default=1)
        parser.add_argument('--no_vis', action='store_true')
        parser.add_argument('--top_k', type=int, default=0)
        parser.add_argument('--shallow', type=str, default='None')
        # parser.add_argument('--num_component', type=int, default=8)
        parser.add_argument('--repeat', type=int, default=1)
        parser.add_argument('--rank', type=int, default=3)
        parser.add_argument('--rank_dim0', type=int, default=3)
        parser.add_argument('--rank_dim1', type=int, default=4)
        parser.add_argument('--rank_dim2', type=int, default=5)
        parser.add_argument('--dropout_rate', type=float, default=0.0)
        parser.add_argument('--remove_loss', type=str,
                            choices=['None', 'sparsity', 'entropy', 'laplacian', 'truth'],
                            default='None')
        parser.add_argument('--interpolation', action='store_true')
        parser.add_argument('--gaussian', action='store_true')
        parser.add_argument('--use_partial', action='store_true')
        parser.add_argument('--dataset_path', type=str, default='dataset')
        parser.add_argument('--train', action='store_true')
        parser.add_argument('--cross_val', action='store_true')
        parser.add_argument('--save_explanation', action='store_true')
        parser.add_argument('--threshold', type=int, default=300, help='threshold for shared explanation graph')
        parser.add_argument('--checkpoint_path', type=str)
        parser.add_argument('--model_logger', action='store_true', help='whether to log a model or not')
        parser.add_argument('--api_key', type=str, help='api key for comet_ml logger')
        parser.add_argument('--project_name', type=str, help='project name for comet_ml logger')

        args = parser.parse_args()
        
        if args.model_logger is not None: #TODO: fix all experiment calls for self.experiment
            self.experiment = Experiment(
                api_key=args.api_key,
                project_name=args.project_name
            )
        else:
            self.experiment = None

        if args.enable_nni:
            args = ModifiedArgs(args, nni.get_next_parameter())

        if os.path.exists(args.save_result):
            os.remove(args.save_result)

        # Report multiple hyperparameters using a dictionary:
        hyper_params = {
            'lr': args.lr,
            'initial_epochs': args.initial_epochs,
            'explainer_epochs': args.explainer_epochs,
            'tuning_epochs': args.tuning_epochs,
            'train_batch_size': args.train_batch_size,
            'test_batch_size': args.test_batch_size,
            'dropout': args.dropout,
            'hidden_dim': args.hidden_dim,
            'n_GNN_layers': args.n_GNN_layers,
            'n_MLP_layers': args.n_MLP_layers
        }
        if self.experiment is not None:
            self.experiment.log_parameters(hyper_params)

            
            
        # load datasets
        dataset_mapping = dict(HIV="datasets/New_Node_AAL90.txt",
                               BP="datasets/New_Node_Brodmann82.txt",
                               PPMI="datasets/New_Node_PPMI.txt",
                               PPMI_balanced="datasets/New_Node_PPMI.txt",
                               Schiza="/home/src/datasets/Schiza.txt")
        txt_name = dataset_mapping.get(args.dataset_name, None)
        node_labels = load_cluster_info_from_txt(txt_name)
        dataset, _, y = load_data_singleview(args, args.dataset_path, args.modality, node_labels)
        train_set, test_set, train_y, test_y = train_test_split(dataset, y, test_size=0.2, \
                                                                shuffle=False, random_state=args.seed)
        node_atts = load_txt(txt_name)
        num_features = dataset[0].x.shape[1]

        # init model
        seed_everything(args.seed)  # use random seed for each run

        if args.cross_val:
            for key in hyper_params.keys():
                print(key, getattr(args, key))
            self.run_cv(args, train_set, train_y, node_atts, node_labels)
            
        if args.train:
            model = build_model(args, device, num_features)
            print('Training 80/20 \n')
            
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            train_loader = DataLoader(train_set, batch_size=args.train_batch_size, shuffle=False)
            test_loader = DataLoader(test_set, batch_size=args.test_batch_size, shuffle=False)

            test_micro, test_auc, test_macro = self.train_and_evaluate(model, train_loader, test_loader,
                                                                       optimizer, device, args, is_tuning=False)
            torch.save(model.state_dict(), './model.pt')

            
        if args.explain:
            if args.checkpoint_path is not None:
                model = build_model(args, device, num_features)
                optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
                model.load_state_dict(torch.load(f'{args.checkpoint_path}/model.pt'))
        
            if args.save_explanation:
                explainer_model, node_feat_mask, edge_mask, \
                msk_train_loader, msk_test_loader = self.explain(model, train_set,test_set, node_labels, node_atts, \
                                                             optimizer, device, args, return_explainer=True, \
                                                             return_loaders=True)

                np.save('./node_feat_mask', node_feat_mask.detach().cpu().numpy())
                sz = int(np.sqrt(edge_mask.shape[0]))
                edge_mask = edge_mask.view(sz, sz).detach().cpu().numpy()
                np.save('./edge_mask', edge_mask)
                
                edges_thresholded = explainer_model.denoise_graph(edge_mask, 0, threshold_num=args.threshold)
                np.savetxt(f"./fig/explainer_{args.dataset_name}_seed{args.seed}_thresh{args.threshold}_sharedmask.edge", \
                           edges_thresholded, delimiter='\t')
                for idx in [0, 1, 5, 6]:
                    explainer_model.plot_explanations(msk_train_loader.dataset[idx], None, args.dataset_name, \
                                                      node_feat_mask, node_atts, name=str(idx), seed=args.seed)
        

def count_degree(data: numpy.ndarray):  # data: (sample, node, node)
    count = numpy.zeros((data.shape[1], data.shape[1]))
    for i in range(data.shape[1]):
        count[i, :] = numpy.sum(data[:, i, :] != 0, axis=0)


if __name__ == '__main__':
    print(device)
    MainExplainer().main()
