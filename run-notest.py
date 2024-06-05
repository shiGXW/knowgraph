import os

from helper import *
from B_data.data_loader import *
# sys.path.append('./')
from C_models.models import *
import pandas as pd


class Runner(object):

    def load_data(self):
        """
        Reading in raw triples and converts it into a standard format.

        Parameters
        ----------
        self.p.dataset:         Takes in the name of the dataset (FB15k-237)

        Returns
        -------
        self.ent2id:            Entity to unique identifier mapping
        self.id2rel:            Inverse mapping of self.ent2id
        self.rel2id:            Relation to unique identifier mapping
        self.num_ent:           Number of entities in the Knowledge graph
        self.num_rel:           Number of relations in the Knowledge graph
        self.embed_dim:         Embedding dimension used
        self.data['train']:     Stores the triples corresponding to training dataset
        self.data['valid']:     Stores the triples corresponding to validation dataset
        self.data['test']:      Stores the triples corresponding to test dataset
        self.data_iter:		The dataloader for different data splits

        """

        ent_set, rel_set = OrderedSet(), OrderedSet()
        for split in ['train', 'valid']:
            for line in open(self.p.data_dir + '{}/all/{}.txt'.format(self.p.dataset, split)):
                sub, rel, obj = map(str.lower, line.strip().split('\t'))
                ent_set.add(sub)
                rel_set.add(rel)
                ent_set.add(obj)

        self.ent2id = {ent: idx for idx, ent in enumerate(ent_set)}
        self.rel2id = {rel: idx for idx, rel in enumerate(rel_set)}
        self.rel2id.update({rel + '_reverse': idx + len(self.rel2id) for idx, rel in enumerate(rel_set)})

        self.id2ent = {idx: ent for ent, idx in self.ent2id.items()}
        self.id2rel = {idx: rel for rel, idx in self.rel2id.items()}

        self.p.num_ent = len(self.ent2id)
        self.p.num_rel = len(self.rel2id) // 2
        self.p.embed_dim = self.p.k_w * self.p.k_h if self.p.embed_dim is None else self.p.embed_dim

        self.data = ddict(list)
        sr2o = ddict(set)

        for split in ['train', 'valid']:
            for line in open(self.p.data_dir + '{}/all/{}.txt'.format(self.p.dataset, split)):
                sub, rel, obj = map(str.lower, line.strip().split('\t'))
                sub, rel, obj = self.ent2id[sub], self.rel2id[rel], self.ent2id[obj]
                self.data[split].append((sub, rel, obj))

                if split == 'train':
                    sr2o[(sub, rel)].add(obj)
                    sr2o[(obj, rel + self.p.num_rel)].add(sub)
            # 将数据打乱
            random.shuffle(self.data[split])

        self.data = dict(self.data)

        self.sr2o = {k: list(v) for k, v in sr2o.items()}
        for split in ['valid']:
            for sub, rel, obj in self.data[split]:
                sr2o[(sub, rel)].add(obj)
                sr2o[(obj, rel + self.p.num_rel)].add(sub)

        self.sr2o_all = {k: list(v) for k, v in sr2o.items()}
        self.triples = ddict(list)

        for (sub, rel), obj in self.sr2o.items():
            self.triples['train'].append({'triple': (sub, rel, -1), 'label': self.sr2o[(sub, rel)], 'sub_samp': 1})

        for split in ['valid']:
            for sub, rel, obj in self.data[split]:
                rel_inv = rel + self.p.num_rel
                self.triples['{}_{}'.format(split, 'tail')].append(
                    {'triple': (sub, rel, obj), 'label': self.sr2o_all[(sub, rel)]})
                self.triples['{}_{}'.format(split, 'head')].append(
                    {'triple': (obj, rel_inv, sub), 'label': self.sr2o_all[(obj, rel_inv)]})

        self.triples = dict(self.triples)

        def get_data_loader(dataset_class, split, batch_size, shuffle=True):
            return DataLoader(
                dataset_class(self.triples[split], self.p),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=max(0, self.p.num_workers),
                collate_fn=dataset_class.collate_fn
            )

        self.data_iter = {
            'train': get_data_loader(TrainDataset, 'train', self.p.batch_size),
            'valid_head': get_data_loader(TestDataset, 'valid_head', self.p.batch_size),
            'valid_tail': get_data_loader(TestDataset, 'valid_tail', self.p.batch_size),
        }

        self.edge_index, self.edge_type = self.construct_adj()

    def construct_adj(self):
        """
        Constructor of the runner class

        Parameters
        ----------

        Returns
        -------
        Constructs the adjacency matrix for GCN

        """
        edge_index, edge_type = [], []

        for sub, rel, obj in self.data['train']:
            edge_index.append((sub, obj))
            edge_type.append(rel)

        # Adding inverse edges
        for sub, rel, obj in self.data['train']:
            edge_index.append((obj, sub))
            edge_type.append(rel + self.p.num_rel)

        edge_index = torch.LongTensor(edge_index).to(self.device).t()
        edge_type = torch.LongTensor(edge_type).to(self.device)

        return edge_index, edge_type

    def __init__(self, params):
        """
        Constructor of the runner class

        Parameters
        ----------
        params:         List of hyper-parameters of the model

        Returns
        -------
        Creates computational graph and optimizer

        """
        self.p = params
        self.logger = get_logger(self.p.name, self.p.log_dir, os.path.abspath(self.p.config_dir))

        self.logger.info(vars(self.p))

        if self.p.gpu != '-1' and torch.cuda.is_available():
            self.device = torch.device('cuda')
            torch.cuda.set_rng_state(torch.cuda.get_rng_state())
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device('cpu')

        self.load_data()
        self.logger.info(f"num_ent: {self.p.num_ent}")
        self.logger.info(f"num_rel: {self.p.num_rel}")
        self.logger.info(f"num_train: {len(self.data['train'])}")
        self.logger.info(f"num_valid: {len(self.data['valid'])}")
        self.model = self.add_model(self.p.model, self.p.score_func)
        self.optimizer = self.add_optimizer(self.model.parameters())

        # 学习率率的变化策略
        if self.p.lr_scheduler == True:
            self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,
                                                                     T_max=20,
                                                                     eta_min=0.0000001,
                                                                     last_epoch=-1)
        else:
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                     milestones=args.lr_steps,
                                                                     gamma=args.lr_gamma)

    def add_model(self, model, score_func):
        """
        Creates the computational graph

        Parameters
        ----------
        model_name:     Contains the model name to be created

        Returns
        -------
        Creates the computational graph for model and initializes it

        """
        model_name = '{}_{}'.format(model, score_func)

        if model_name.lower() == 'compgcn_transe':
            model = CompGCN_TransE(self.edge_index, self.edge_type, params=self.p)
        elif model_name.lower() == 'compgcn_distmult':
            model = CompGCN_DistMult(self.edge_index, self.edge_type, params=self.p)
        elif model_name.lower() == 'compgcn_conve':
            model = CompGCN_ConvE(self.edge_index, self.edge_type, params=self.p)
        elif model_name.lower() == 'compgcn_transformer':
            model = CompGCN_Transformer(self.edge_index, self.edge_type, params=self.p)
        else:
            raise NotImplementedError

        model.to(self.device)
        return model

    def add_optimizer(self, parameters):
        """
        Creates an optimizer for training the parameters

        Parameters
        ----------
        parameters:         The parameters of the model

        Returns
        -------
        Returns an optimizer for learning the parameters of the model

        """
        return torch.optim.Adam(parameters, lr=self.p.lr, weight_decay=self.p.l2)

    def read_batch(self, batch, split):
        """
        Function to read a batch of data and move the tensors in batch to CPU/GPU

        Parameters
        ----------
        batch: 		the batch to process
        split: (string) If split == 'train', 'valid' or 'test' split


        Returns
        -------
        Head, Relation, Tails, labels
        """
        if split == 'train':
            triple, label = [_.to(self.device) for _ in batch]
            return triple[:, 0], triple[:, 1], triple[:, 2], label
        else:
            triple, label = [_.to(self.device) for _ in batch]
            return triple[:, 0], triple[:, 1], triple[:, 2], label

    def save_model(self, save_path):
        """
        Function to save a model. It saves the model parameters, best validation scores,
        best epoch corresponding to best validation, state of the optimizer and all arguments for the run.

        Parameters
        ----------
        save_path: path where the model is saved

        Returns
        -------
        """
        state = {
            'state_dict': self.model.state_dict(),
            'best_val': self.best_val,
            'best_epoch': self.best_epoch,
            'optimizer': self.optimizer.state_dict(),
            'lr_scheduler': self.lr_scheduler.state_dict(),
            'args': vars(self.p)
        }
        torch.save(state, save_path)

    def load_model(self, load_path):
        """
        Function to load a saved model

        Parameters
        ----------
        load_path: path to the saved model

        Returns
        -------
        """
        state = torch.load(load_path)
        state_dict = state['state_dict']
        self.best_val = state['best_val']
        self.best_val_mrr = self.best_val['mrr']

        self.model.load_state_dict(state_dict)
        self.optimizer.load_state_dict(state['optimizer'])
        self.lr_scheduler.load_state_dict(state['lr_scheduler'])

    def save_csv(self, sub_total, rel_total, obj_total, target_pred_total, label_total, results, rel_flag):
        sub_total = np.array(sub_total, dtype=np.int32)
        rel_total = np.array(rel_total, dtype=np.int32)
        obj_total = np.array(obj_total, dtype=np.int32)
        label_total = np.array(label_total, dtype=np.int32)
        rel_index = np.where(rel_total == rel_flag)
        rel_list = ["industry", "areacode", "HW", "waste", "material", "product", "HWwaste"]
        # sub_total, rel_total, obj_total, label_total
        # 头，关系，尾，头与关系对应的全部尾的索引(第二位为 entid )
        # sub_total = [ 5962, 98, 1246, 5063, 6696, ... ]
        # rel_total = [ 13, 26, 7, 2, 19, ... ]
        # obj_total = [ 2292, 1276, 3175, 3010, 339, ... ]
        # target_pred_total = [ 0.00005, 0.00128, 0.72369, 0.16455, 0.00001, ... ]
        # label_total = [ [ 0, 27 ], [ 0, 32 ], [ 0, 75 ], [ 0, 92 ], [ 0, 119 ], ... ]
        export_info = {
            "enter": OrderedSet(),
            "enterid": OrderedSet(),
            "pred_" + rel_list[rel_flag]: [],
            "true_" + rel_list[rel_flag]: [],
            rel_list[rel_flag] + "_accuracy": [],
        }
        # id_labels_total = { "0": [ 27, 32, 75, 92, 119 ], ... }
        # 获取 md5 值对应的真实id
        with open(os.path.join(self.p.data_dir + self.p.dataset, 'id_md5_dict.json'), 'r') as json_file:
            id_md5_dict = json.loads(str(json_file.read()))
            enterid2id_dict = dict(zip(id_md5_dict.values(), id_md5_dict.keys()))
        # 获取 id 对应企业名
        with open(os.path.join(self.p.data_dir + self.p.dataset, 'id_enterprise_dict.json'), 'r') as json_file:
            id2enterprise_dict = json.loads(str(json_file.read()))

        pred_industry_total = {}
        true_industry_total = {}
        for item in rel_index[0]:
            # 写入 Set
            export_info["enter"].add(id2enterprise_dict[enterid2id_dict[self.id2ent[sub_total[item]]]])
            export_info["enterid"].add(self.id2ent[sub_total[item]])
            # 处理 pred_industry
            # 预测阈值
            # print(target_pred_total[item])
            if target_pred_total[item] >= self.p.accuracy_th:
                if sub_total[item] in pred_industry_total.keys():
                    pred_industry_total[str(sub_total[item])].append(self.id2ent[obj_total[item]])
                else:
                    pred_industry_total[str(sub_total[item])] = [self.id2ent[obj_total[item]]]
        # 处理 true_industry
        for item in range(len(label_total)):
            sr2o_rel_flag = self.sr2o_all.get((label_total[item][0], rel_flag), [])
            if label_total[item][1] in sr2o_rel_flag:
                if self.id2ent[int(label_total[item][0])] in true_industry_total.keys():
                    true_industry_total[self.id2ent[int(label_total[item][0])]].append(self.id2ent[int(label_total[item][1])])
                else:
                    true_industry_total[self.id2ent[int(label_total[item][0])]] = [self.id2ent[int(label_total[item][1])]]

        for item in export_info["enterid"]:
            true_list = list(set(true_industry_total.get(item, [])))
            pred_list = list(set(pred_industry_total.get(item, [])))
            export_info["true_" + rel_list[rel_flag]].append('、'.join(true_list))
            export_info["pred_" + rel_list[rel_flag]].append('、'.join(pred_list))
            export_info[rel_list[rel_flag] + "_accuracy"].append(
                len(set(true_list) & set(pred_list)) / (len(true_list) if len(true_list) > len(pred_list) else len(pred_list))
            )
        # pred_targets = torch.where(pred_targets, label, -1).to("cpu")
        # true_targets = label.to("cpu")
        # # 计算准确率
        # torch.eq(pred_targets, true_targets).sum().float().item()
        # 导出到 csv 文件
        time_flag = time.strftime('%Y_%m_%d') + '_' + time.strftime('%H_%M_%S') + '_' + str(results['mrr'])
        pd.DataFrame({
            0: list(export_info["enter"]),
            1: list(export_info["enterid"]),
            2: export_info["pred_industry"],
            3: export_info["true_industry"]
        }).to_csv(
            f'{args.csv_dir}/{time_flag}.csv', sep='\t', index=False, header=False
        )

    def evaluate(self, split, epoch):
        """
        Function to evaluate the model on validation or test set

        Parameters
        ----------
        split: (string) If split == 'valid' then evaluate on the validation set, else the test set
        epoch: (int) Current epoch count

        Returns
        -------
        resutls:			The evaluation results containing the following:
            results['mr']:         	Average of ranks_left and ranks_right
            results['mrr']:         Mean Reciprocal Rank
            results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

        """
        # left_results = self.predict(split=split, mode='tail_batch')
        left_results, sub_total, rel_total, obj_total, target_pred_total, label_total = self.predict(split=split, mode='tail_batch')
        right_results = self.predict(split=split, mode='head_batch')
        results = get_combined_results(left_results, right_results)
        self.logger.info('[Epoch {} {}]: MRR: Tail : {:.5f}, Head : {:.5f}, Avg : {:.5f}\n'.format(
            epoch, split, results['left_mrr'], results['right_mrr'], results['mrr']))
        self.save_csv(sub_total, rel_total, obj_total, target_pred_total, label_total, results, rel_flag=0)
        return results

    def predict(self, split='valid', mode='tail_batch'):
        """
        Function to run model evaluation for a given mode

        Parameters
        ----------
        split: (string) 	If split == 'valid' then evaluate on the validation set, else the test set
        mode: (string):		Can be 'head_batch' or 'tail_batch'

        Returns
        -------
        resutls:			The evaluation results containing the following:
            results['mr']:         	Average of ranks_left and ranks_right
            results['mrr']:         Mean Reciprocal Rank
            results['hits@k']:      Probability of getting the correct preodiction in top-k ranks based on predicted score

        """
        self.model.eval()

        sub_total = np.empty(0)
        rel_total = np.empty(0)
        obj_total = np.empty(0)
        # 所预测 obj 概率
        target_pred_total = np.empty(0)
        label_total = np.empty((0, 2))

        with torch.no_grad():
            results = {}
            train_iter = iter(self.data_iter['{}_{}'.format(split, mode.split('_')[0])])

            for step, batch in enumerate(train_iter):
                sub, rel, obj, label = self.read_batch(batch, split)
                pred = self.model.forward(sub, rel)
                b_range = torch.arange(pred.size()[0], device=self.device)
                # 预测概率——对应于每一个 obj
                target_pred = pred[b_range, obj]
                if mode == 'tail_batch':
                    # 存入三元组
                    sub_total = np.concatenate((sub_total, sub.cpu()), axis=0)
                    rel_total = np.concatenate((rel_total, rel.cpu()), axis=0)
                    obj_total = np.concatenate((obj_total, obj.cpu()), axis=0)
                    # 找出tensor中非零的元素的索引，存入label_total
                    label_nonzero = label.nonzero().cpu()
                    label_nonzero = np.array(label_nonzero, dtype=np.int32)
                    for index in range(label_nonzero.shape[0]):
                        label_nonzero[index, 0] = sub[label_nonzero[index, 0]]
                        label_nonzero[index, 1] = label_nonzero[index, 1]
                    label_total = np.concatenate((label_total, label_nonzero), axis=0)
                    target_pred_total = np.concatenate((target_pred_total, target_pred.cpu()), axis=0)
                pred = torch.where(label.bool(), -torch.ones_like(pred) * 10000000, pred)
                pred[b_range, obj] = target_pred
                ranks = 1 + torch.argsort(
                    torch.argsort(pred, dim=1, descending=True), dim=1, descending=False
                )[b_range, obj]

                ranks = ranks.float()
                results['count'] = torch.numel(ranks) + results.get('count', 0.0)
                results['mr'] = torch.sum(ranks).item() + results.get('mr', 0.0)
                results['mrr'] = torch.sum(1.0 / ranks).item() + results.get('mrr', 0.0)
                for k in range(10):
                    results['hits@{}'.format(k + 1)] = torch.numel(ranks[ranks <= (k + 1)]) + results.get('hits@{}'.format(k + 1), 0.0)

                if step % self.p.print_fre == 0:
                    self.logger.info('[{}| {} Step {}]\t{}'.format(split.title(), mode.title(), step, results['mrr']))
        if mode == 'tail_batch':
            return results, sub_total, rel_total, obj_total, target_pred_total, label_total
        else:
            return results

    def run_epoch(self, epoch):
        """
        Function to run one epoch of training

        Parameters
        ----------
        epoch: current epoch count

        Returns
        -------
        loss: The loss value after the completion of one epoch
        """
        self.model.train()
        losses = []
        train_iter = iter(self.data_iter['train'])

        for step, batch in enumerate(train_iter):
            self.optimizer.zero_grad()
            sub, rel, obj, label = self.read_batch(batch, 'train')

            pred = self.model.forward(sub, rel)
            loss = self.model.loss(pred, label)

            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

            if step % self.p.print_fre == 0:
                self.logger.info('[Epoch:{}| {}]: Train Loss:{:.5f}'.format(epoch, step, np.mean(losses)))

        loss = np.mean(losses)
        self.logger.info('[Epoch {} Loss]:  Training Loss:{:.5f}\n'.format(epoch, loss))
        return loss

    def fit(self):
        """
        Function to run training and evaluation of model

        Parameters
        ----------

        Returns
        -------
        """
        self.best_val_mrr, self.best_val, self.best_epoch, val_mrr = 0., {}, 0, 0.
        save_path = os.path.join(self.p.checkpoints_dir, self.p.name)

        if self.p.restore:
            self.load_model(save_path)
            self.logger.info('Successfully Loaded previous model')

        kill_cnt = 0
        for epoch in range(self.p.max_epochs):
            train_loss = self.run_epoch(epoch)
            val_results = self.evaluate('valid', epoch)

            # if val_results['mrr'] > self.best_val_mrr:
            if val_results['left_mrr'] > self.best_val_mrr:
                self.best_val = val_results
                self.best_val_mrr = val_results['left_mrr']
                self.best_epoch = epoch
                self.save_model(save_path)
                kill_cnt = 0
            else:
                kill_cnt += 1
                if kill_cnt % 10 == 0 and self.p.gamma > 5:
                    self.p.gamma -= 5
                    self.logger.info('Gamma decay on saturation, updated value of gamma: {}'.format(self.p.gamma))
                if kill_cnt > 500:
                    self.logger.info("Early Stopping!!")
                    break

            self.logger.info(
                '[Epoch {}]: lr:{:.9f}, Training Loss: {:.5f}, Tail MRR: {:.5f}, H@10: {:.5f}, H@1: {:.5f}\n'.format(
                    epoch, self.optimizer.param_groups[0]["lr"], train_loss,
                    val_results['left_mrr'], val_results['hits@10'], val_results['hits@1'])
            )

            # 更新学习率
            self.lr_scheduler.step()


if __name__ == '__main__':
    # /home/GXW/code/pytorch/CompGCN-master
    parser = argparse.ArgumentParser(description='Parser For Arguments',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-name', default='testrun', help='Set run name for saving/restoring models')
    parser.add_argument('-data', dest='dataset', default='knowgraph/max/', help='Dataset to use, default: FB15k-237, knowgraph/max/')
    parser.add_argument('-model', dest='model', default='CompGCN', help='Model Name')
    parser.add_argument('-score_func', dest='score_func', default='Transformer', help='Score Function for Link prediction')
    parser.add_argument('-opn', dest='opn', default='corr', help='Composition Operation to be used in CompGCN：sub, mult, corr')

    # ConvE：896
    # Transformer：1664、2048
    parser.add_argument('-batch', dest='batch_size', default=2048, type=int, help='Batch size: 256, 896, 1664')
    parser.add_argument('-print', dest='print_fre', default=10, type=int, help='Printing frequency：Batch num')
    parser.add_argument('-gamma', type=float, default=40.0, help='Margin')
    parser.add_argument('-gpu', type=str, default='0', help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0, 1')
    parser.add_argument('-epoch', dest='max_epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('-l2', type=float, default=0.0, help='L2 Regularization for Optimizer')
    # ConvE：0.01
    # Transformer：0.001
    parser.add_argument('-lr', type=float, default=0.001, help='Starting Learning Rate')
    # 针对torch.optim.lr_scheduler
    parser.add_argument('-lr_scheduler', dest='lr_scheduler', type=bool, default=False, help='Label Smoothing')
    # MultiStepLR的参数
    # 设置学习率降低的epoch位置
    # ConvE：[16, 22], [6, 16], [50, 100, 150, 200, 300], [100, 200, 350, 400, 450]
    # Transformer：[10, 20, 30, 40, 50], [100, 200, 300, 400, 500]
    parser.add_argument('--lr-steps', type=int, default=[300, 425, 450, 475], nargs='+',
                        help='decrease lr every step-size epochs')
    # 学习率衰减的倍数
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='decrease lr by a factor of lr-gamma')

    parser.add_argument('-lbl_smooth', dest='lbl_smooth', type=float, default=0.1, help='Label Smoothing')
    parser.add_argument('-num_workers', type=int, default=0, help='Number of processes to construct batches')
    parser.add_argument('-seed', dest='seed', default=7588, type=int, help='Seed for randomization')
    parser.add_argument('-accuracy_th', type=float, default=0.5, help='预测阈值')

    parser.add_argument('-restore', dest='restore', action='store_true', help='Restore from the previously saved model')
    parser.add_argument('-bias', dest='bias', action='store_true', help='Whether to use bias in the model')

    parser.add_argument('-num_bases', dest='num_bases', default=7, type=int,
                        help='Number of basis relation vectors to use: -1, 7')
    parser.add_argument('-init_dim', dest='init_dim', default=200, type=int,
                        help='Initial dimension size for entities and relations')
    parser.add_argument('-embed_dim', dest='embed_dim', default=200, type=int,
                        help='Embedding dimension to give as input to score function')
    parser.add_argument('-gcn_dim', dest='gcn_dim', default=150, type=int, help='Number of hidden units in GCN')
    parser.add_argument('-gcn_layer', dest='gcn_layer', default=2, type=int, help='Number of GCN Layers to use')
    parser.add_argument('-gcn_drop', dest='dropout', default=0.1, type=float, help='Dropout to use in GCN Layer')
    parser.add_argument('-hid_drop', dest='hid_drop', default=0.1, type=float, help='Dropout after GCN')

    # ConvE specific hyperparameters
    parser.add_argument('-hid_drop2', dest='hid_drop2', default=0.3, type=float, help='ConvE: Hidden dropout')
    parser.add_argument('-feat_drop', dest='feat_drop', default=0.3, type=float, help='ConvE: Feature Dropout')
    parser.add_argument('-k_w', dest='k_w', default=10, type=int, help='ConvE: k_w')
    parser.add_argument('-k_h', dest='k_h', default=20, type=int, help='ConvE: k_h')
    parser.add_argument('-num_filt', dest='num_filt', default=200, type=int,
                        help='ConvE: Number of filters in convolution')
    parser.add_argument('-ker_sz', dest='ker_sz', default=5, type=int, help='ConvE: Kernel size to use')

    # Transformer specific hyperparameters
    parser.add_argument('-T_layers', dest='T_layers', default=2, type=int, help='Transformer: layers')
    parser.add_argument('-T_num_heads', dest='T_num_heads', default=4, type=int, help='Transformer: num_heads')
    parser.add_argument('-T_num_hidden', dest='T_num_hidden', default=2048, type=int, help='Transformer: num_hidden')
    parser.add_argument('-T_hid_drop2', dest='T_hid_drop2', default=0.3, type=float, help='Transformer: Hidden dropout')
    parser.add_argument('-T_hid_drop', dest='T_hid_drop', default=0.1, type=float, help='Dropout after GCN')
    parser.add_argument('-T_feat_drop', dest='T_feat_drop', default=0.3, type=float, help='Transformer: Feature Dropout')
    parser.add_argument('-T_pooling', dest='T_pooling', default='avg', type=str, help='Transformer: min / avg / concat')
    parser.add_argument('-T_flat', dest='T_flat', default=15, type=int, help='Transformer: num_heads')
    parser.add_argument('-T_positional', dest='T_positional', default=True, type=bool, help='Transformer: positional')

    parser.add_argument('-logdir', dest='log_dir', default='./D_export/log/', help='Log directory')
    parser.add_argument('-checkpointsdir', dest='checkpoints_dir', default='./D_export/checkpoints/', help='Log directory')
    parser.add_argument('-csvdir', dest='csv_dir', default='./D_export/log/', help='csv directory')
    parser.add_argument('-configdir', dest='config_dir', default='./A_config/', help='Config directory')
    parser.add_argument('-datadir', dest='data_dir', default='./B_data/datasets/', help='data directory')
    args = parser.parse_args()

    if not args.restore: args.name = args.name + '_' + time.strftime('%Y_%m_%d') + '_' + time.strftime('%H_%M_%S')

    set_gpu(args.gpu)

    # 创建 csv 存储位置
    args.csv_dir = args.csv_dir + args.name + "_csv"
    os.mkdir(args.csv_dir)

    args.seed = np.random.randint(1000, 100000)

    # 固定所有的随机种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model = Runner(args)
    model.fit()
