import torch
from . import networks
from os.path import join
from util.util import seg_accuracy, print_network


class ClassifierModel:
    """ Class for training Model weights

    :args opt: structure containing configuration params
    e.g.,
    --dataset_mode -> classification / segmentation)
    --arch -> network type
    """
    def __init__(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.is_train = opt.is_train
        self.device = torch.device('cuda:{}'.format(self.gpu_ids[0])) if self.gpu_ids else torch.device('cpu')
        self.save_dir = join(opt.checkpoints_dir, opt.name)
        self.optimizer = None
        self.edge_features = None
        self.labels = None
        self.mesh = None
        self.soft_label = None
        self.loss = None
        self.save_dir = join(opt.checkpoints_dir, opt.name)
        self.testacc_log = join(self.save_dir, 'testacc_full_log.txt')
        # MAKE THIS CSV


        #
        self.nclasses = opt.nclasses

        # load/define networks
        self.net = networks.define_classifier(opt.input_nc, opt.ncf, opt.ninput_edges, opt.nclasses, opt,
                                              self.gpu_ids, opt.arch, opt.init_type, opt.init_gain)
        self.net.train(self.is_train)
        self.criterion = networks.define_loss(opt).to(self.device)

        if self.is_train:
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.scheduler = networks.get_scheduler(self.optimizer, opt)
            print_network(self.net)

        if not self.is_train or opt.continue_train:
            self.load_network(opt.which_epoch)

    def set_input(self, data):
        input_edge_features = torch.from_numpy(data['edge_features']).float()
        if self.opt.dataset_mode == "regression":
            labels = torch.from_numpy(data['label']).float()
        else:
            labels = torch.from_numpy(data['label']).long()
        # set inputs
        self.edge_features = input_edge_features.to(self.device).requires_grad_(self.is_train)
        self.labels = labels.to(self.device)
        self.mesh = data['mesh']
        # Adding path
        self.path = data['path']
        if self.opt.dataset_mode == 'segmentation' and not self.is_train:
            self.soft_label = torch.from_numpy(data['soft_label'])


    def forward(self):
        out = self.net(self.edge_features, self.mesh)
        return out

    def backward(self, out):
        if self.opt.dataset_mode == "regression":
            # print("OUTPUT", out.view(-1), "LABEL", self.labels)
            self.loss = self.criterion(out.view(-1), self.labels)
        else:
            self.loss = self.criterion(out, self.labels)
        self.loss.backward()

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        out = self.forward()
        self.backward(out)
        self.optimizer.step()


##################

    def load_network(self, which_epoch):
        """load model from disk"""
        save_filename = '%s_net.pth' % which_epoch
        load_path = join(self.save_dir, save_filename)
        net = self.net
        if isinstance(net, torch.nn.DataParallel):
            net = net.module
        print('loading the model from %s' % load_path)
        # PyTorch newer than 0.4 (e.g., built from
        # GitHub source), you can remove str() on self.device
        state_dict = torch.load(load_path, map_location=str(self.device))
        if hasattr(state_dict, '_metadata'):
            del state_dict._metadata
        net.load_state_dict(state_dict)


    def save_network(self, which_epoch):
        """save model to disk"""
        save_filename = '%s_net.pth' % (which_epoch)
        save_path = join(self.save_dir, save_filename)
        if len(self.gpu_ids) > 0 and torch.cuda.is_available():
            torch.save(self.net.module.cpu().state_dict(), save_path)
            self.net.cuda(self.gpu_ids[0])
        else:
            torch.save(self.net.cpu().state_dict(), save_path)

    def update_learning_rate(self):
        """update learning rate (called once every epoch)"""
        self.scheduler.step()
        lr = self.optimizer.param_groups[0]['lr']
        print('learning rate = %.7f' % lr)

    def test(self):
        """tests model
        returns: number correct and total number
        """
        with torch.no_grad():
            out = self.forward()
            # compute number of correct
            if self.opt.dataset_mode == 'regression':
                pred_class = out.view(-1)
            else:
                pred_class = out.data.max(1)[1]
            #pred_class = self.forward()
            # compute number of correct
            #pred_class.reshape((pred_class.shape[0]))
            label_class = self.labels
            self.export_segmentation(pred_class.cpu())

            patient_id = self.path[-1][36:-4]

            print('-------')
            print('Patient ID:\t', patient_id)
            print('Predicted:\t', pred_class.item())
            print('Label:\t\t', label_class.item())
            correct = self.get_accuracy(pred_class, label_class)
            print('Abs Error:\t', correct.item())

            # MAKE FILE FOR EACH EPOCH
            if not self.opt.is_train:
                with open(self.testacc_log, "a") as log_file:
                    log_file.write(f"{patient_id},{pred_class.item()},{label_class.item()},{correct.item()}")
        return correct, len(label_class)

    def get_accuracy(self, pred, labels):
        """computes accuracy for classification / segmentation """
        if self.opt.dataset_mode == 'classification':
            correct = pred.eq(labels).sum()
        elif self.opt.dataset_mode == 'segmentation':
            correct = seg_accuracy(pred, self.soft_label, self.mesh)
        elif self.opt.dataset_mode == 'regression':
            mean_abs_err = torch.nn.functional.l1_loss(pred, labels, reduction='mean')
            correct = mean_abs_err
        return correct

    def export_segmentation(self, pred_seg):
        if self.opt.dataset_mode == 'segmentation':
            for meshi, mesh in enumerate(self.mesh):
                mesh.export_segments(pred_seg[meshi, :])
