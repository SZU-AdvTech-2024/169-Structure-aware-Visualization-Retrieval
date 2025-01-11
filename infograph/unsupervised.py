import torch as th
import numpy as np
import dgl
import shutil
import pickle
from dataset import VizMLPlusDataset
from dgl.dataloading import GraphDataLoader

from model import InfoGraph
from collections import Counter

from time import time

import argparse


def argument():
    parser = argparse.ArgumentParser(description='InfoGraph')
    # data source params
    parser.add_argument('--input_path', type=str, default='../full_VizML+/train_graph', help='Path of Datasets')

    # training params
    parser.add_argument('--gpu', type=int, default=-1, help='GPU index, default:-1, using CPU.')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs.')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size.')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate.')
    parser.add_argument('--log_interval', type=int, default=1, help='Interval between two evaluations.')
    parser.add_argument('--save_interval', type=int, default=5, help='Interval between two checkpoints.')
    parser.add_argument('--checkpoint_path', default="./cp/", help='Path to save checkpoints')
    parser.add_argument('--embedding_path', default="./embedding/", help='Path to save embeddings')
    parser.add_argument('--model_path', default="./model_best.pth.tar", help='path to load model for testing')
    parser.add_argument('--mode', default="train", help='train or test')

    # model params
    parser.add_argument('--n_layers', type=int, default=3, help='Number of graph convolution layers before each pooling.')
    parser.add_argument('--hid_dim', type=int, default=32, help='Hidden layer dimensionalities.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and th.cuda.is_available():
        args.device = 'cuda:{}'.format(args.gpu)
    else:
        args.device = 'cpu'

    return args

    
def collate(samples):
    ''' collate function for building graph dataloader'''
    
    graphs, labels = map(list, zip(*samples))
    # print(f"Graphs: {len(graphs)}; Labels: {Counter(labels)}")

    # generate batched graphs and labels
    batched_graph = dgl.batch(graphs)
    batched_labels = th.tensor(labels)

    n_graphs = len(graphs)
    graph_id = th.arange(n_graphs)
    graph_id = dgl.broadcast_nodes(batched_graph, graph_id)

    batched_graph.ndata['graph_id'] = graph_id

    return batched_graph, batched_labels

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    th.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

def save_embedding(embedding, is_best, dtype='torch', filename='embedding.pkl'):
    with open(filename, 'wb') as f:
        if dtype == 'torch':
            pickle.dump(embedding.numpy(), f)
        else:
            pickle.dump(embedding, f)
    if is_best:
        shutil.copyfile(filename, 'embedding_best.pkl')

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


if __name__ == '__main__':

    # Step 1: Prepare graph data   ===================================== #
    args = argument()
    print(args)

    # load dataset from dgl.data.GINDataset
    dataset = VizMLPlusDataset(args.input_path)

    # get graphs and labels
    graphs, labels = map(list, zip(*dataset))
    
    # generate a full-graph with all examples for evaluation
    wholegraph = dgl.batch(graphs)
    wholegraph.ndata['vector'] = wholegraph.ndata['vector'].to(th.float32)

    # create dataloader for batch training
    # dataloader = GraphDataLoader(dataset,
    #                              batch_size=args.batch_size,
    #                              collate_fn=collate,
    #                              drop_last=False,
    #                              shuffle=True)

    # in_dim = wholegraph.ndata['vector'].shape[1]

    # # Step 2: Create model =================================================================== #
    # model = InfoGraph(in_dim, args.hid_dim, args.n_layers)
    # model = model.to(args.device)

    if args.mode == "train":
        dataloader = GraphDataLoader(dataset,
                                 batch_size=args.batch_size,
                                 collate_fn=collate,
                                 drop_last=False,
                                 shuffle=True)

        in_dim = wholegraph.ndata['vector'].shape[1]

        # Step 2: Create model =================================================================== #
        model = InfoGraph(in_dim, args.hid_dim, args.n_layers)
        model = model.to(args.device)
        # Step 3: Create training components ===================================================== #
        del wholegraph
        optimizer = th.optim.Adam(model.parameters(), lr=args.lr)
    
        ''' Evaluate the initialized embeddings '''
        ''' using logistic regression and SVM(non-linear) '''
        # print('logreg {:4f}, svc {:4f}'.format(res[0], res[1]))
        
        best_logreg = 0
        best_logreg_epoch = 0
        best_svc = 0
        best_svc_epoch = 0

    # Step 4: training epochs =============================================================== #
        end = time()
        for epoch in range(args.epochs):
            loss_all = 0
            model.train()
            # count = 0
            t_start = time()
        
            for i, (graph, label) in enumerate(dataloader):

                batch_time = AverageMeter('Time', ':6.3f')
                data_time = AverageMeter('Data', ':6.3f')
                losses = AverageMeter('Loss', ':.4f')
                progress = ProgressMeter(
                    len(dataloader),
                    [batch_time, data_time, losses],
                    prefix="Epoch: [{}]".format(epoch))
                
                graph = graph.to(args.device)
                feat = graph.ndata['vector']
                # print(feat.shape)
                graph_id = graph.ndata['graph_id']
                
                n_graph = label.shape[0]

                
        
                optimizer.zero_grad()
                loss = model(graph, feat, graph_id)
                loss.backward()
                optimizer.step()
                loss_all += loss.item()

                losses.update(loss.item(), args.batch_size)

                batch_time.update(time() - end)
                end = time()

                if i % 10 == 0:
                    progress.display(i)
        
            print('Epoch {}, Loss {:.4f}'.format(epoch, loss_all))

            if (epoch+1) % args.save_interval == 0:
                    save_checkpoint({
                        'epoch': epoch+1,
                        'state_dict': model.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                    }, is_best=False, filename= args.checkpoint_path + 'checkpoint_{:04d}.pth.tar'.format(epoch+1))
                    
                    # save_embedding(emb, is_best=False, filename= args.embedding_path + 'embedding_{:04d}.pkl'.format(epoch))

                    

        print('Training End')
        print('best logreg {:4f} ,best svc {:4f}'.format(best_logreg, best_svc))
    
    else:
        dataloader = GraphDataLoader(dataset,
                                 batch_size=args.batch_size,
                                 collate_fn=collate,
                                 drop_last=False,
                                 shuffle=False)

        in_dim = wholegraph.ndata['vector'].shape[1]
        model = InfoGraph(in_dim, args.hid_dim, args.n_layers)
        model = model.to(args.device)

        loaded_model = th.load(args.model_path)

        model.load_state_dict(loaded_model['state_dict'])
        model.eval()

        list_embedding = []
        for i, (graph, label) in enumerate(dataloader):

                batch_time = AverageMeter('Time', ':6.3f')
                data_time = AverageMeter('Data', ':6.3f')
                losses = AverageMeter('Loss', ':.4f')
                progress = ProgressMeter(
                    len(dataloader),
                    [batch_time, data_time, losses],
                    prefix="Evaluation: ")
                
                graph = graph.to(args.device)
                feat = graph.ndata['vector']
                # print(feat.shape)
                graph_id = graph.ndata['graph_id']
                
                n_graph = label.shape[0]

                emb = model.get_embedding(graph, feat).cpu().detach().numpy()
                list_embedding.append(emb)

        np_embeddings = np.concatenate(list_embedding, axis=0)

        dataset_name = 'train' if 'train' in args.input_path else 'test'

        save_embedding(np_embeddings, is_best=False, dtype='numpy',\
         filename= args.embedding_path + 'checkpoint_{:04d}_embedding_{}.pkl'.format(loaded_model['epoch'], dataset_name))