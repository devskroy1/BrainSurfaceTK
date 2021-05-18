
import os
import os.path as osp

PATH_TO_ROOT = osp.join(osp.dirname(osp.realpath(__file__)), '..', '..')
import sys

sys.path.append(PATH_TO_ROOT)

import math
import subprocess

import dgl
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models.gNNs.data_utils import BrainNetworkDataset
from models.gNNs.networks import BasicGCNRegressor
from models.gNNs.gcnCamVtk import add_node_saliency_scores_to_vtk


def collate(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    subjects, graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return np.array(subjects).reshape(len(graphs), -1), batched_graph, torch.tensor(labels).view(len(graphs), -1)


def denorm_target_f(target, dataset):
    # used to unstandardise the target (to get the scan age of the patient)
    return (target.cpu() * dataset.targets_std) + dataset.targets_mu


def str_to_bool(x):
    if x == "True":
        return True
    elif x == "False":
        return False
    else:
        raise ValueError("Expected True or False for featureless.")

#For native surface vtks
def str_to_features(x):
    if x.lower() == "all".lower():
        return ('corrected_thickness', 'initial_thickness', 'curvature', 'sulcal_depth', 'roi')
    elif x.lower() == "some".lower():
        return ('corrected_thickness', 'curvature', 'sulcal_depth')
    elif x.lower() == "None".lower():
        return None

#For aligned surface vtks
# def str_to_features(x):
#     if x.lower() == "all".lower():
#         return ('corrected_thickness', 'initial_thickness', 'curvature', 'sulcal_depth', 'roi')
#     elif x.lower() == "some".lower():
#         return ('corrThickness', 'Curvature', 'Sulc')
#     elif x.lower() == "None".lower():
#         return None

#For native surface vtks
def features_to_str(x):
    if x == ('corrected_thickness', 'initial_thickness', 'curvature', 'sulcal_depth', 'roi'):
        return "all"
    elif x == ('corrected_thickness', 'curvature', 'sulcal_depth'):
        return "some"
    elif x == None:
        return "None"

#For aligned surface vtks
# def features_to_str(x):
#     if x == ('corrected_thickness', 'initial_thickness', 'curvature', 'sulcal_depth', 'roi'):
#         return "all"
#     elif x == ('corrThickness', 'Curvature', 'Sulc'):
#         return "some"
#     elif x == None:
#         return "None"

def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    # Dataset/Dataloader Args
    parser.add_argument("load_path", help="location where files are stored (.vtk)/(.vtp)", type=str)
    parser.add_argument("featureless", help="don't include features? True or False", type=str_to_bool)
    parser.add_argument("features", help="""'None', 'some' or 'all' 
    (edit the features_to_str function to customise tailor the names of the features to your needs)""",
                        type=str_to_features)
    parser.add_argument("--meta_data_file_path", help="tsv file containing patient data", type=str, default="/vol/biomedic/users/aa16914/shared/data/dhcp_neonatal_brain/combined.tsv")
                        # default="/vol/biomedic2/aa16914/shared/MScAI_brain_surface/data/meta_data.tsv")
    parser.add_argument("--pickle_split_filepath", help="split file", type=str, default=None)
                        # default="/vol/bitbucket/cnw119/neodeepbrain/models/gNNs/names_06152020_noCrashSubs.pk")
    parser.add_argument("--ds_max_workers", help="max_workers for building dataset", type=int, default=8)
    parser.add_argument("--dl_max_workers", help="max_workers for dataloader", type=int, default=4)
    parser.add_argument("--save_path", help="where to store the dataset files", type=str, default="../tmp")

    # Training Args
    parser.add_argument("--max_epochs", help="max epochs", type=int, default=300)
    parser.add_argument("--batch_size", help="batch size", type=int, default=64)
    parser.add_argument("--lr", help="lr", type=float, default=8e-4)
    parser.add_argument("--T_max", help="T_max", type=int, default=10)
    parser.add_argument("--eta_min", help="eta_min", type=float, default=1e-6)

    # Results Args
    parser.add_argument("--results", help="where to store results", type=str, default="./results")

    args = parser.parse_args()

    args.save_path = os.path.join(args.save_path, f"features-{features_to_str(args.features)}_dataset")

    args.experiment_name = f"GCN-features-{features_to_str(args.features)}"

    args.experiment_folder = os.path.join(args.results, args.experiment_name)

    if not os.path.exists(args.experiment_folder):
        os.makedirs(args.experiment_folder)

    print("Using files from: ", args.load_path)
    print("Data saved in: ", args.save_path)
    print("Results stored in: ", args.experiment_folder)

    return args


def get_dataloaders(args):
    train_dataset = BrainNetworkDataset(args.load_path, args.meta_data_file_path, save_path=args.save_path,
                                        max_workers=args.ds_max_workers,
                                        dataset="train", index_split_pickle_fp=args.pickle_split_filepath,
                                        features=args.features)

    val_dataset = BrainNetworkDataset(args.load_path, args.meta_data_file_path, save_path=args.save_path,
                                      max_workers=args.ds_max_workers,
                                      dataset="val", features=args.features)

    test_dataset = BrainNetworkDataset(args.load_path, args.meta_data_file_path, save_path=args.save_path,
                                       max_workers=args.ds_max_workers,
                                       dataset="test", features=args.features)

    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate,
                          num_workers=args.dl_max_workers)
    val_dl = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
                        num_workers=args.dl_max_workers)
    test_dl = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate,
                         num_workers=args.dl_max_workers)
    print("Dataloaders created")
    return train_dl, val_dl, test_dl, train_dataset, val_dataset, test_dataset

#Train with basic saliency scores
def train(model, train_dl, train_ds, loss_function, diff_func, denorm_target, optimizer, scheduler, device):
    model.train()
    train_epoch_loss = 0
    train_epoch_error = 0.
    train_epoch_worst_diff = 0.
    train_total_size = 0
    for iter, (subjects, bg, batch_labels) in enumerate(train_dl):
        optimizer.zero_grad()

        # print("Train subjects")
        # print(subjects)

        bg = bg.to(device)

        graphs = dgl.unbatch(bg)
        first_graph = graphs[0]
        first_graph_node_features = first_graph.ndata["features"].to(device)

        bg_node_features = bg.ndata["features"].to(device)
        batch_labels = batch_labels.to(device)

        #prediction = model(bg, bg_node_features, is_training=True)

        predictions, _ = model(graph=bg, features=bg_node_features, is_training=True)

        loss = loss_function(predictions, batch_labels)
        loss.backward()
        optimizer.step()

        if (epoch == args.max_epochs - 1) and (iter < 6):
            _, saliency_scores = model(graph=first_graph, features=first_graph_node_features, is_training=True)
            # saliency_scores = cam[:, :num_nodes_first_graph]
            # print("saliency scores shape")
            # print(saliency_scores.shape)
            # Append saliency scores to VTK only for the first subject in the batch
            # print("Before calling add_node_saliency_scores_to_vtk()")
            add_node_saliency_scores_to_vtk(saliency_scores=saliency_scores, vtk_root=args.load_path,
                                            subject=subjects[0][0])

        with torch.no_grad():
            train_diff = diff_func(denorm_target(predictions, train_ds),
                                   denorm_target(batch_labels, train_ds))  # shape: (batch_size, 1)
            train_epoch_error += train_diff.sum().detach().item()
            worst_diff = torch.max(train_diff).detach().item()
            if worst_diff > train_epoch_worst_diff:
                train_epoch_worst_diff = worst_diff
        train_epoch_loss += loss.detach().item()
        train_total_size += len(batch_labels)

    # train_epoch_loss = train_epoch_loss / (iter + 1)
    train_epoch_loss /= (iter + 1)  # Calculate mean sum batch loss over this epoch MSELoss
    train_epoch_error /= train_total_size  # Calculate mean L1 error over all the training data over this epoch

    scheduler.step()

    return train_epoch_loss, train_epoch_error, train_epoch_worst_diff

#Original train()
# def train(model, train_dl, train_ds, loss_function, diff_func, denorm_target, optimizer, scheduler, device):
#     model.train()
#     train_epoch_loss = 0
#     train_epoch_error = 0.
#     train_epoch_worst_diff = 0.
#     train_total_size = 0
#     for iter, (subjects, bg, batch_labels) in enumerate(train_dl):
#         optimizer.zero_grad()
#
#         # print("Train subjects")
#         # print(subjects)
#
#         bg = bg.to(device)
#         bg_node_features = bg.ndata["features"].to(device)
#         batch_labels = batch_labels.to(device)
#         prediction = model(bg, bg_node_features, is_training=True)
#         loss = loss_function(prediction, batch_labels)
#         loss.backward()
#         optimizer.step()
#
#         with torch.no_grad():
#             train_diff = diff_func(denorm_target(prediction, train_ds),
#                                    denorm_target(batch_labels, train_ds))  # shape: (batch_size, 1)
#             train_epoch_error += train_diff.sum().detach().item()
#             worst_diff = torch.max(train_diff).detach().item()
#             if worst_diff > train_epoch_worst_diff:
#                 train_epoch_worst_diff = worst_diff
#         train_epoch_loss += loss.detach().item()
#         train_total_size += len(batch_labels)
#
#     # train_epoch_loss = train_epoch_loss / (iter + 1)
#     train_epoch_loss /= (iter + 1)  # Calculate mean sum batch loss over this epoch MSELoss
#     train_epoch_error /= train_total_size  # Calculate mean L1 error over all the training data over this epoch
#
#     scheduler.step()
#
#     return train_epoch_loss, train_epoch_error, train_epoch_worst_diff

#Pop salcy map train()
# def train(model, train_dl, train_ds, loss_function, diff_func, denorm_target, optimizer, scheduler, device):
#     model.train()
#     train_epoch_loss = 0
#     train_epoch_error = 0.
#     train_epoch_worst_diff = 0.
#     train_total_size = 0
#
#     k = 1000
#     sum_topk_indices = torch.zeros(k, device=device)
#     sum_saliency_scores = torch.zeros(k, device=device)
#     num_subjects = 0
#     for iter, (subjects, bg, batch_labels) in enumerate(train_dl):
#
#         torch.cuda.empty_cache()
#         optimizer.zero_grad()
#
#         bg = bg.to(device)
#         graphs = dgl.unbatch(bg)
#         first_graph = graphs[0]
#         bg_node_features = bg.ndata["features"].to(device)
#
#         if (epoch == args.max_epochs - 1):
#             first_graph_node_features = first_graph.ndata["features"].to(device)
#             num_nodes_first_graph = first_graph_node_features.size(0)
#
#         batch_labels = batch_labels.to(device)
#
#         graph, prediction = model(graph=bg, features=bg_node_features, is_training=True)
#
#         print("len(graphs)")
#         print(len(graphs))
#         if (epoch == args.max_epochs - 1):
#             for g in range(len(graphs)):
#                 graph = graphs[g]
#                 bg_node_features = graph.ndata["features"].to(device)
#                 graph, preds = model(graph=graph, features=bg_node_features, is_training=False)
#
#                 num_subjects += 1
#                 top_nodes_indices = dgl.topk_nodes(graph, 'saliency_score', k)
#                 # print("top_nodes_indices")
#                 # print(top_nodes_indices)
#                 # print("top_nodes_indices shape")
#                 # print(top_nodes_indices.shape)
#                 saliency_scores = graph.ndata['saliency_score']
#                 print("saliency_scores train")
#                 print(saliency_scores)
#                 highest_saliency_scores = torch.index_select(saliency_scores, 0, top_nodes_indices[0])
#                 # print("highest_saliency_scores shape")
#                 # print(highest_saliency_scores.shape)
#                 print("highest_saliency_scores train")
#                 print(highest_saliency_scores)
#                 sum_topk_indices += top_nodes_indices[0]
#                 sum_saliency_scores += highest_saliency_scores
#
#         loss = loss_function(prediction, batch_labels)
#         loss.backward()
#         optimizer.step()
#
#         with torch.no_grad():
#             train_diff = diff_func(denorm_target(prediction, train_ds),
#                                    denorm_target(batch_labels, train_ds))  # shape: (batch_size, 1)
#             train_epoch_error += train_diff.sum().detach().item()
#             worst_diff = torch.max(train_diff).detach().item()
#             if worst_diff > train_epoch_worst_diff:
#                 train_epoch_worst_diff = worst_diff
#         train_epoch_loss += loss.detach().item()
#         train_total_size += len(batch_labels)
#
#         break
#     if (epoch == args.max_epochs - 1):
#
#         # print("sum_topk_indices")
#         # print(sum_topk_indices)
#         # print("sum_saliency_scores")
#         # print(sum_saliency_scores)
#
#         pop_saliency_indices = sum_topk_indices // num_subjects
#         pop_saliency_indices = pop_saliency_indices.long()
#         pop_saliency_scores = sum_saliency_scores * (1 / num_subjects)
#
#         # print("pop_saliency_indices")
#         # print(pop_saliency_indices)
#         print("pop_saliency_scores train")
#         print(pop_saliency_scores)
#         # print("num_nodes_first_graph")
#         # print(num_nodes_first_graph)
#
#         saliency_scores = torch.zeros(num_nodes_first_graph, device=device)
#         for i in range(k):
#             pop_saliency_index = pop_saliency_indices[i]
#             saliency_scores[pop_saliency_index] = pop_saliency_scores[i]
#
#         add_node_saliency_scores_to_vtk(saliency_scores=saliency_scores,
#                                         vtk_root=args.load_path, subject=subjects[0][0])
#
#     # train_epoch_loss = train_epoch_loss / (iter + 1)
#     train_epoch_loss /= (iter + 1)  # Calculate mean sum batch loss over this epoch MSELoss
#     train_epoch_error /= train_total_size  # Calculate mean L1 error over all the training data over this epoch
#
#     scheduler.step()
#
#     return train_epoch_loss, train_epoch_error, train_epoch_worst_diff

# Basic saliency scores
# def evaluate(model, dl, ds, loss_function, diff_func, denorm_target_f, device, val):
#     with torch.no_grad():
#         model.eval()
#         epoch_loss = 0
#         epoch_error = 0.
#         total_size = 0
#         epoch_max_diff = 0.
#         batch_subjects = list()
#         batch_preds = list()
#         batch_targets = list()
#         batch_diffs = list()
#         # batch_size = args.batch_size
#         # print("About to enter evaluate for loop")
#         for iter, (subjects, bg, batch_labels) in enumerate(dl):
#             # print("About to evaluate on new batch of subject graphs")
#             # print("iter")
#             # print(iter)
#             # print("batch_labels")
#             # print(batch_labels)
#             # print("batch_labels shape")
#             # print(batch_labels.shape)
#             # bg stands for batch graph
#             #if (iter > 5) and (epoch == 1) and not val:
#             if (iter > 5) and (epoch == args.max_epochs - 1) and not val:
#                 break
#             bg = bg.to(device)
#             # get node feature
#             graphs = dgl.unbatch(bg)
#             first_graph = graphs[0]
#             bg_node_features = bg.ndata["features"].to(device)
#             #total_num_nodes = bg_node_features.size(0)
#
#             first_graph_node_features = first_graph.ndata["features"].to(device)
#             num_nodes_first_graph = first_graph_node_features.size(0)
#
#             #num_nodes_per_graph = total_num_nodes // batch_size
#
#             batch_labels = batch_labels.to(device)
#
#             predictions, _ = model(graph=bg, features=bg_node_features, is_training=False)
#
#             #i = 0
#             #for subject in subjects:
#                 # print("i")
#                 # print(i)
#             #saliency_scores = cam[:, i * num_nodes_per_graph:(i + 1) * num_nodes_per_graph]
#             #if (epoch == 1) and not val:
#             if (epoch == args.max_epochs - 1) and not val:
#                 _, saliency_scores = model(graph=first_graph, features=first_graph_node_features, is_training=False)
#                 # saliency_scores = cam[:, :num_nodes_first_graph]
#                 # print("saliency scores shape")
#                 # print(saliency_scores.shape)
#                 #Append saliency scores to VTK only for the first subject in the batch
#                 # print("Before calling add_node_saliency_scores_to_vtk()")
#                 add_node_saliency_scores_to_vtk(saliency_scores=saliency_scores, vtk_root=args.load_path,
#                                             subject=subjects[0][0])
#                 # print("After calling add_node_saliency_scores_to_vtk()")
#             # print("After calling add_node_saliency_scores_to_vtk()")
#              #   i += 1
#
#             loss = loss_function(predictions, batch_labels)
#             # if (epoch == args.max_epochs - 1) and not val:
#             #     if loss.item() > 10:
#             #         print("Poorly extracted subject surfaces")
#             #         print(subjects)
#             # print("After calling loss function")
#             diff = diff_func(denorm_target_f(predictions, ds),
#                              denorm_target_f(batch_labels, ds))
#             epoch_error += diff.sum().item()
#             # Identify max difference
#             max_diff = torch.max(diff).item()
#             if max_diff > epoch_max_diff:
#                 epoch_max_diff = max_diff
#             epoch_loss += loss.item()
#
#             # Store
#             batch_subjects.append(subjects)
#             batch_preds.append(predictions.cpu())
#             batch_targets.append(batch_labels.cpu())
#             batch_diffs.append(diff.cpu())
#
#             total_size += len(batch_labels)
#         # print("After exiting for loop iterating over subject batches")
#         epoch_loss /= (iter + 1)
#         epoch_error /= total_size
#
#         all_subjects = np.concatenate(batch_subjects)
#         all_preds = denorm_target_f(torch.cat(batch_preds), ds)
#         all_targets = denorm_target_f(torch.cat(batch_targets), ds)
#         all_diffs = torch.cat(batch_diffs)
#
#         # csv_material = np.concatenate((all_subjects, all_preds.numpy(), all_targets.numpy(), all_diffs.numpy()),
#         #                               axis=-1)
#
#     return epoch_loss, epoch_error, torch.max(all_diffs).item()

# Population level saliency map method (with aligned surfaces)
# def evaluate(model, dl, ds, loss_function, diff_func, denorm_target_f, device, val):
#     with torch.no_grad():
#         model.eval()
#         epoch_loss = 0
#         epoch_error = 0.
#         total_size = 0
#         epoch_max_diff = 0.
#         batch_subjects = list()
#         batch_preds = list()
#         batch_targets = list()
#         batch_diffs = list()
#         # batch_size = args.batch_size
#         # print("About to enter evaluate for loop")
#         # print("len test data loader")
#         # print(len(dl))
#         k = 1000
#         sum_topk_indices = torch.zeros(k, device=device)
#         sum_saliency_scores = torch.zeros(k, device=device)
#         num_subjects = 0
#         for iter, (subjects, bg, batch_labels) in enumerate(dl):
#             torch.cuda.empty_cache()
#             # print("Eval subjects")
#             # print(subjects)
#             # print("batch labels")
#             # print(batch_labels)
#             # print("About to evaluate on new batch of subject graphs")
#             # print("iter")
#             # print(iter)
#             # print("bg")
#             # print(bg)
#             # bg stands for batch graph
#             bg = bg.to(device)
#             # get node feature
#             graphs = dgl.unbatch(bg)
#             # print("num graphs in batch")
#             # print(len(graphs))
#             first_graph = graphs[0]
#
#             #bg_node_features = first_graph.ndata["features"].to(device)
#             bg_node_features = bg.ndata["features"].to(device)
#             # print("bg.ndata[features] first dims")
#             # print(bg.ndata["features"].size(0))
#
#             #total_num_nodes = bg_node_features.size(0)
#             #if (epoch == 1) and not val:
#             if (epoch == args.max_epochs - 1) and not val:
#                 first_graph_node_features = first_graph.ndata["features"].to(device)
#                 num_nodes_first_graph = first_graph_node_features.size(0)
#
#             #num_nodes_per_graph = total_num_nodes // batch_size
#
#             batch_labels = batch_labels.to(device)
#
#             # print("first_graph")
#             # print(first_graph)
#             # print("bg_node_features")
#             # print(bg_node_features)
#
#             graph, predictions = model(graph=bg, features=bg_node_features, is_training=False)
#
#             #k = 100
#
#             # print("top_nodes")
#             # print(top_nodes)
#             # print("top_nodes shape")
#             # print(top_nodes.shape)
#
#             # for b in range(args.batch_size):
#             #     graph = graphs[b]
#             #     bg_node_features = graph.ndata["features"].to(device)
#             #     # predictions, cam = model(graph=first_graph, features=bg_node_features, is_training=False)
#             #     graph, predictions, cam = model(graph=graph, features=bg_node_features, is_training=False)
#             #
#             #     print("cam")
#             #     print(cam)
#             #     print("predictions")
#             #     print(predictions)
#
#             #i = 0
#             #for subject in subjects:
#                 # print("i")
#                 # print(i)
#             #saliency_scores = cam[:, i * num_nodes_per_graph:(i + 1) * num_nodes_per_graph]
#
#             #if (epoch == 1) and not val:
#             if (epoch == args.max_epochs - 1) and not val:
#                 for g in range(len(graphs)):
#                     graph = graphs[g]
#                     bg_node_features = graph.ndata["features"].to(device)
#                     graph, preds = model(graph=graph, features=bg_node_features, is_training=False)
#
#                     num_subjects += 1
#                     top_nodes_indices = dgl.topk_nodes(graph, 'saliency_score', k)
#                     # print("top_nodes_indices")
#                     # print(top_nodes_indices)
#                     # print("top_nodes_indices shape")
#                     # print(top_nodes_indices.shape)
#                     saliency_scores = graph.ndata['saliency_score']
#                     # print("saliency_scores evaluate")
#                     # print(saliency_scores)
#                     highest_saliency_scores = torch.index_select(saliency_scores, 0, top_nodes_indices[0])
#                     # print("highest_saliency_scores shape")
#                     # print(highest_saliency_scores.shape)
#                     # print("highest_saliency_scores evaluate")
#                     # print(highest_saliency_scores)
#                     sum_topk_indices += top_nodes_indices[0]
#                     sum_saliency_scores += highest_saliency_scores
#
#                 # saliency_scores = cam[:, :num_nodes_first_graph]
#
#                 # print("saliency scores shape")
#                 # print(saliency_scores.shape)
#                 #Append saliency scores to VTK only for the first subject in the batch
#                 # print("Before calling add_node_saliency_scores_to_vtk()")
#
#                 # add_node_saliency_scores_to_vtk(saliency_scores=saliency_scores, vtk_root=args.load_path,
#                 #                             subject=subjects[0][0])
#
#                 # print("After calling add_node_saliency_scores_to_vtk()")
#             # print("After calling add_node_saliency_scores_to_vtk()")
#              #   i += 1
#
#             loss = loss_function(predictions, batch_labels)
#
#             # print("After calling loss function")
#             diff = diff_func(denorm_target_f(predictions, ds),
#                              denorm_target_f(batch_labels, ds))
#             epoch_error += diff.sum().item()
#             # Identify max difference
#             max_diff = torch.max(diff).item()
#             if max_diff > epoch_max_diff:
#                 epoch_max_diff = max_diff
#             epoch_loss += loss.item()
#
#             # Store
#             batch_subjects.append(subjects)
#             batch_preds.append(predictions.cpu())
#             batch_targets.append(batch_labels.cpu())
#             batch_diffs.append(diff.cpu())
#
#             total_size += len(batch_labels)
#             break
#
#         #if (epoch == 1) and not val:
#         if (epoch == args.max_epochs - 1) and not val:
#
#             # print("sum_topk_indices")
#             # print(sum_topk_indices)
#             # print("sum_saliency_scores")
#             # print(sum_saliency_scores)
#
#             pop_saliency_indices = sum_topk_indices // num_subjects
#             pop_saliency_indices = pop_saliency_indices.long()
#             pop_saliency_scores = sum_saliency_scores * (1 / num_subjects)
#
#             # print("pop_saliency_indices")
#             # print(pop_saliency_indices)
#             # print("pop_saliency_scores evaluate")
#             # print(pop_saliency_scores)
#             # print("num_nodes_first_graph")
#             # print(num_nodes_first_graph)
#
#             saliency_scores = torch.zeros(num_nodes_first_graph, device=device)
#             for i in range(k):
#                 pop_saliency_index = pop_saliency_indices[i]
#                 saliency_scores[pop_saliency_index] = pop_saliency_scores[i]
#
#             add_node_saliency_scores_to_vtk(saliency_scores=saliency_scores,
#                                             vtk_root=args.load_path, subject=subjects[0][0])
#         # print("After exiting for loop iterating over subject batches")
#         epoch_loss /= (iter + 1)
#         epoch_error /= total_size
#
#         all_subjects = np.concatenate(batch_subjects)
#         all_preds = denorm_target_f(torch.cat(batch_preds), ds)
#         all_targets = denorm_target_f(torch.cat(batch_targets), ds)
#         all_diffs = torch.cat(batch_diffs)
#
#         # csv_material = np.concatenate((all_subjects, all_preds.numpy(), all_targets.numpy(), all_diffs.numpy()),
#         #                               axis=-1)
#
#     return epoch_loss, epoch_error, torch.max(all_diffs).item()

#Original evaluate()
def evaluate(model, dl, ds, loss_function, diff_func, denorm_target_f, device):
    with torch.no_grad():
        model.eval()
        epoch_loss = 0
        epoch_error = 0.
        total_size = 0
        epoch_max_diff = 0.
        batch_subjects = list()
        batch_preds = list()
        batch_targets = list()
        batch_diffs = list()
        for iter, (subjects, bg, batch_labels) in enumerate(dl):
            # bg stands for batch graph
            bg = bg.to(device)
            # get node feature
            bg_node_features = bg.ndata["features"].to(device)
            batch_labels = batch_labels.to(device)

            predictions = model(bg, bg_node_features)
            loss = loss_function(predictions, batch_labels)

            diff = diff_func(denorm_target_f(predictions, ds),
                             denorm_target_f(batch_labels, ds))
            epoch_error += diff.sum().item()
            # Identify max difference
            max_diff = torch.max(diff).item()
            if max_diff > epoch_max_diff:
                epoch_max_diff = max_diff
            epoch_loss += loss.item()

            # Store
            batch_subjects.append(subjects)
            batch_preds.append(predictions.cpu())
            batch_targets.append(batch_labels.cpu())
            batch_diffs.append(diff.cpu())

            total_size += len(batch_labels)

        epoch_loss /= (iter + 1)
        epoch_error /= total_size

        all_subjects = np.concatenate(batch_subjects)
        all_preds = denorm_target_f(torch.cat(batch_preds), ds)
        all_targets = denorm_target_f(torch.cat(batch_targets), ds)
        all_diffs = torch.cat(batch_diffs)

        # csv_material = np.concatenate((all_subjects, all_preds.numpy(), all_targets.numpy(), all_diffs.numpy()),
        #                               axis=-1)

    return epoch_loss, epoch_error, torch.max(all_diffs).item()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def update_best_model(model, validation_loss, best_validation_loss, args):
    if validation_loss < best_validation_loss:
        torch.save(model, f=os.path.join(args.experiment_folder, "best_model"))
        return validation_loss
    else:
        return best_validation_loss


def update_writer(writer, train_epoch_loss, val_epoch_loss, test_epoch_loss, train_epoch_error, val_epoch_error,
                  test_epoch_error, train_epoch_max_diff, val_epoch_max_diff, test_epoch_max_diff, epoch):
    writer.add_scalar("Loss/Train", train_epoch_loss, epoch)
    writer.add_scalar("Loss/Val", val_epoch_loss, epoch)
    writer.add_scalar("Loss/Test", test_epoch_loss, epoch)
    writer.add_scalar("Error/Train", train_epoch_error, epoch)
    writer.add_scalar("Error/Val", val_epoch_error, epoch)
    writer.add_scalar("Error/Test", test_epoch_error, epoch)
    writer.add_scalar("Max Error/Train", train_epoch_max_diff, epoch)
    writer.add_scalar("Max Error/Val", val_epoch_max_diff, epoch)
    writer.add_scalar("Max Error/Test", test_epoch_max_diff, epoch)


def record_csv_material(fp, data):
    if os.path.exists(fp):
        ndarray = np.load(fp)
        ndarray = np.concatenate((ndarray, data.reshape(1, *data.shape)))
    else:
        ndarray = data.reshape(1, *data.shape)
    np.save(file=fp, arr=ndarray)


def get_gpu_memory_map():
    """Get the current gpu usage.
    Gotta love StackOverflow
    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


if __name__ == "__main__":
    # TODO MAKE NOT HARD CODED FOR IMPERIAL (I think this has been done but ping cnw119/cemlyn007 just in case)

    args = get_args()

    val_log_fp = os.path.join(args.experiment_folder, "val_log")
    test_log_fp = os.path.join(args.experiment_folder, "test_log")

    train_dl, val_dl, test_dl, train_ds, val_ds, test_ds = get_dataloaders(args)

    writer = SummaryWriter(comment=f"-{args.experiment_name}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Create model
    print("Creating Model")
    # print("len(args.features)")
    # print(len(args.features))
    model = BasicGCNRegressor(3 + len(args.features), 256, 1, device)  # 5 features in a node, 256 in the hidden, 1 output (age)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max, eta_min=args.eta_min)

    model = model.to(device)

    print(model)
    print(f"Model is on: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print("Total number of parameters: ", count_parameters(model))

    loss_function = nn.MSELoss(reduction="sum")

    diff_func = nn.L1Loss(reduction="none")

    best_val_loss = math.inf

    print("Starting")
    for epoch in range(args.max_epochs):
        # Train
        train_epoch_loss, train_epoch_error, train_epoch_max_diff = train(model, train_dl, train_ds, loss_function,
                                                                          diff_func, denorm_target_f, optimizer,
                                                                          scheduler, device)

        # Val
        val_epoch_loss, val_epoch_error, val_epoch_max_diff = evaluate(model, val_dl, val_ds,
                                                                                         loss_function,
                                                                                         diff_func, denorm_target_f,
                                                                                         device, True)
        # Test
        test_epoch_loss, test_epoch_error, test_epoch_max_diff = evaluate(model, test_dl, test_ds,
                                                                                             loss_function,
                                                                                             diff_func, denorm_target_f,
                                                                                             device, False)

        # Record to TensorBoard
        update_writer(writer, train_epoch_loss, val_epoch_loss, test_epoch_loss, train_epoch_error, val_epoch_error,
                      test_epoch_error, train_epoch_max_diff, val_epoch_max_diff, test_epoch_max_diff, epoch)

        # Record material to be converted to csv later
        # record_csv_material(val_log_fp + ".npy", val_csv_material)
        # record_csv_material(test_log_fp + ".npy", test_csv_material)

        # Save model
        update_best_model(model, val_epoch_loss, best_val_loss, args)
        torch.save(model, os.path.join(args.experiment_folder, "curr_model"))

        print('Epoch {}, train_loss {:.4f}, test_loss {:.4f}'.format(epoch, train_epoch_loss, test_epoch_loss))

    mem = get_gpu_memory_map()

    print(mem)

    with open(os.path.join(args.experiment_folder, "GPU_mem.txt"), "w") as f:
        f.write(str(mem))