'''
        Run on ImageNet-Dogs by:
        python main.py
        Run on DTD by:
        python main.py --data DTD --n_clusters 47 --epo 100
'''
import os
import sys

sys.path.append('./')

from datetime import datetime
current_date = datetime.now()
formatted_date = current_date.strftime('%m-%d')

import argparse
import json
import torch
import numpy as np
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from model.model import MORSE
from model.sink_distance import SinkhornDistance
from loss.loss_fn import TotalCodingRate
from utils import *
from metrics.clustering import spectral_clustering_metrics
from data_utils import  ImgTextDataset

parser = argparse.ArgumentParser(description='PRO-DSC Training')
parser.add_argument('--desc', type=str, default='exp',
                    help='description of the experiment')

parser.add_argument('--data', type=str, default='ImageNet-Dogs',
                    help='dataset to use')
parser.add_argument('--gamma', type=int, default=150, 
                    help='coeff for expr loss')
parser.add_argument('--seed', type=int, default=42,
                    help='random seed')

parser.add_argument('--hidden_dim', type=int, default=512,
                    help='hidden dimension of the pre_feature layer')
parser.add_argument('--z_dim', type=int, default=128,
                    help='dimension of the learned representation')
parser.add_argument('--n_clusters', type=int, default=15,
                    help='number of subspaces to cluster')
parser.add_argument('--epo', type=int, default=50,
                    help='number of epochs for training')
parser.add_argument('--bs', type=int, default=1024,
                    help='input batch size for training')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 0.0001)')
parser.add_argument('--momo', type=float, default=0.9,
                    help='momentum (default: 0.9)')
parser.add_argument('--wd1', type=float, default=1e-4,
                    help='weight decay (default: 1e-4)')
parser.add_argument('--pieta', type=float, default=0.1,
                    help='hyper-parameter for Sinkhorn projection')
parser.add_argument('--piiter', type=int, default=5,
                    help='hyper-parameter for Sinkhorn projection')
parser.add_argument('--eps', type=float, default=0.1,
                    help='eps squared for total coding rate (default: 0.1)')
parser.add_argument('--validate_every', type=int, default=0,
                    help='optional diagnostic test interval; 0 evaluates only the final epoch')
parser.add_argument('--eval_seed', type=int, default=0,
                    help='fixed random seed for label-independent spectral clustering')
parser.add_argument('--eval_repeats', type=int, default=10,
                    help='number of K-means fits to average, using the same partitions for all metrics')
args = parser.parse_args()

datasets_list = ['cifar-10','cifar-20','imagenet-10','imagenet-dogs','stl-10','dtd','ucf101','imagenet']
assert args.data.lower() in datasets_list, "Only {} are supported".format(','.join(datasets_list))

args.desc = '_'.join(
    [formatted_date, args.data, 'gamma{}'.format(args.gamma), 'seed{}'.format(args.seed), args.desc])
print(args)
#################################################################################################################
dir_name = os.path.join(f'exps/{args.desc}')
writer = init_pipeline(dir_name, args)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Loading features and labels
images_embedding_train = np.load(os.path.join('data/datasets', args.data + "_image_embedding_train.npy"))
images_embedding_train = images_embedding_train / np.linalg.norm(images_embedding_train, axis=1, keepdims=True)
images_embedding_test = np.load(os.path.join('data/datasets', args.data + "_image_embedding_test.npy"))
images_embedding_test = images_embedding_test / np.linalg.norm(images_embedding_test, axis=1, keepdims=True)
nouns_embedding = np.load(os.path.join('data/datasets', args.data + "_MP5iters.npy"))
nouns_embedding = nouns_embedding / np.linalg.norm(nouns_embedding, axis=1, keepdims=True)
labels_train = np.loadtxt(os.path.join('data/datasets', args.data + "_labels_train.txt"))
labels_test = np.loadtxt(os.path.join('data/datasets', args.data + "_labels_test.txt"))

model = MORSE(input_dim=512, hidden_dim=args.hidden_dim, z_dim=args.z_dim).cuda()
dataset_image_train = TensorDataset(torch.from_numpy(images_embedding_train).float())
dataset_text_train = TensorDataset(torch.from_numpy(nouns_embedding).float())
dataset_label_train = TensorDataset(torch.from_numpy(labels_train))
dataset_image_test = TensorDataset(torch.from_numpy(images_embedding_test).float())

dataset = ImgTextDataset(dataset_image_train, dataset_text_train, dataset_label_train)
dataloader_train = DataLoader(dataset, batch_size=args.bs, num_workers=0, shuffle=True, drop_last=True)
dataloader_test = DataLoader(dataset_image_test, batch_size=args.bs, num_workers=0, shuffle=False, drop_last=False)

#### loss of logdet()
logdet_criterion = TotalCodingRate(eps=args.eps)

### optimizer
optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momo, weight_decay=args.wd1, nesterov=False)
scaler = GradScaler()
sink_layer = SinkhornDistance(args.pieta, max_iter=args.piiter)

with tqdm(total=args.epo) as progress_bar:
    for epoch in range(args.epo):
        progress_bar.set_description('Epoch: '+str(epoch)+'/'+str(args.epo))
        model.train()

        for step, (img, text, label, index) in enumerate(dataloader_train):
            img, text, label, index = img[0].float().to(device), text[0].to(device), label[0], index.to(device)
            with autocast(enabled=True):
                img_z, text_z = model(img, text)
                avg_z = 0.5 *(img_z + text_z)
                self_coeff = avg_z @ avg_z.T

                Sign_self_coeff = torch.sign(self_coeff)
                
                ### Sinkhorn projection 
                self_coeff = self_coeff.abs().unsqueeze(0)
                Pi = sink_layer(self_coeff)[0]
                Pi = Pi * Pi.shape[-1]
                self_coeff = Pi[0]

                # eliminate the diagonal value of self_coeff, which fits the constraint of C
                self_coeff = self_coeff - torch.diag(torch.diag(self_coeff))

                loss_exp_img = 0.5 * (torch.linalg.norm(img_z.T - img_z.T @ Sign_self_coeff.mul(self_coeff) )) ** 2 / args.bs # ||Z-ZC||_F loss
                loss_exp_text = 0.5 * (torch.linalg.norm(text_z.T - text_z.T @ Sign_self_coeff.mul(self_coeff) )) ** 2 / args.bs # ||Z-ZC||_F loss
                loss_tcr_img = logdet_criterion(img_z) # logdet() loss
                loss_tcr_text = logdet_criterion(text_z) # logdet() loss

                loss = (loss_tcr_img + loss_tcr_text) + args.gamma * (loss_exp_img + loss_exp_text) 

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
                
            progress_bar.set_postfix(
                tcr_loss_img="{:5.2f}".format(loss_tcr_img.item()),
                tcr_loss_text="{:5.2f}".format(loss_tcr_text.item()),
                exp_loss_img="{:5.2f}".format(loss_exp_img.item()),
                exp_loss_text="{:5.2f}".format(loss_exp_text.item()),
            )
        progress_bar.update(1)

        ### evaluate on test set
        if (args.validate_every > 0 and (epoch + 1) % args.validate_every == 0) or (epoch + 1) == args.epo:
            print('EVAL on TEST SET (final epoch is the reporting checkpoint)')
            model.eval()
            with torch.random.fork_rng(), torch.no_grad():
                z_list = []
                
                for step, x in enumerate(dataloader_test):
                    x = x[0].float().to(device)
                    img_z = model.img_feature(x)
                    img_z = torch.nn.functional.normalize(img_z, 2)
                    z_list.append(img_z)
                    
                z = torch.cat(z_list, dim=0)
                self_coeff = (z @ z.T).abs()

                acc_lst, nmi_lst, ari_lst, pred_lst = spectral_clustering_metrics(
                    self_coeff.detach().cpu().numpy(), args.n_clusters, labels_test,
                    n_init=args.eval_repeats, random_state=args.eval_seed)
                scores = {'ACC': acc_lst, 'NMI': nmi_lst, 'ARI': ari_lst}
                summary = {name: {'mean': float(np.mean(values)),
                                  'std': float(np.std(values, ddof=0))}
                           for name, values in scores.items()}
                for name, values in summary.items():
                    writer.add_scalar(name + '/mean', values['mean'], global_step=epoch)
                    writer.add_scalar(name + '/std', values['std'], global_step=epoch)
                print('epoch {}: {}'.format(epoch + 1, summary))
                np.savez_compressed(
                    os.path.join(dir_name, 'evaluation_epoch_{}.npz'.format(epoch + 1)),
                    labels=labels_test, predictions=np.stack(pred_lst), **scores)
                if epoch + 1 == args.epo:
                    torch.save({'model': model.state_dict(), 'args': vars(args),
                                'epoch': epoch + 1},
                               os.path.join(dir_name, 'checkpoints', 'final.pt'))
                    with open(os.path.join(dir_name, 'evaluation_final.json'), 'w') as f:
                        json.dump({'epoch': epoch + 1, 'training_seed': args.seed,
                                   'evaluation_seed': args.eval_seed,
                                   'evaluation_repeats': args.eval_repeats,
                                   'kmeans_restarts_per_fit': 10,
                                   'selection': 'predefined_final_epoch',
                                   'std_ddof': 0, 'metrics': summary}, f, indent=2)

writer.close()
