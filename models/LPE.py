import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet import ResNet

import numpy as np

class LPE(nn.Module):

    def __init__(self, args, mode=None):
        super().__init__()
        self.mode = mode
        self.args = args

        self.encoder = ResNet(args=args)
        self.encoder_dim = 640
        self.fc = nn.Linear(self.encoder_dim, self.args.num_class) 

        if self.args.is_LPE:
            self.encoder_dim = 640
            self.sem_dim = args.sem_dim
            for i in range(args.n_attr_templet):
                setattr(self,'sem2attr_{}'.format(i),nn.Sequential(nn.Linear(self.sem_dim, self.sem_dim), nn.LeakyReLU(0.1), \
                                                      nn.Linear(self.sem_dim, self.encoder_dim)))
            if self.args.templet_weight == 'sem_generate':
                self.sem2templet_weight = nn.Sequential(nn.Linear(self.sem_dim, self.sem_dim//2), nn.LeakyReLU(0.1), \
                                                    nn.Linear(self.sem_dim//2, args.n_attr_templet), nn.Sigmoid())
                self.templet_weight_bias = nn.Parameter(torch.zeros(args.n_attr_templet), requires_grad=True)
                self.tmpweight_lambda1 = nn.Parameter(torch.FloatTensor(1).fill_(1.0), requires_grad=True)
                self.tmpweight_lambda2 = nn.Parameter(torch.FloatTensor(1).fill_(1.0), requires_grad=True)
            weight_base = torch.FloatTensor(self.args.num_class, args.n_attr_templet, self.encoder_dim).normal_(0.0, np.sqrt(2.0/self.encoder_dim))
            self.weight_base = nn.Parameter(weight_base, requires_grad=True)
            if not self.args.no_transferbase:
                for i in range(args.n_attr_templet):
                    # queryLayer
                    setattr(self,'queryLayer_{}'.format(i),nn.Linear(self.encoder_dim, self.encoder_dim))
                    getattr(self,'queryLayer_{}'.format(i)).weight.data.copy_(
                        torch.eye(self.encoder_dim, self.encoder_dim) + torch.randn(self.encoder_dim, self.encoder_dim)*0.001)
                    getattr(self,'queryLayer_{}'.format(i)).bias.data.zero_()
                    # wkeys
                    wkeys = torch.FloatTensor(self.args.num_class, self.encoder_dim).normal_(0.0, np.sqrt(2.0/self.encoder_dim))
                    setattr(self,'wkeys_{}'.format(i),nn.Parameter(wkeys, requires_grad=True))
                    setattr(self,'scale_att_{}'.format(i),nn.Parameter(torch.FloatTensor(1).fill_(10.0), requires_grad=True))
                self.class_no = torch.tensor(range(self.args.num_class)).cuda()
                self.lamda_novel = nn.Parameter(torch.FloatTensor(args.n_attr_templet).fill_(1.0), requires_grad=True)
                self.lamda_base = nn.Parameter(torch.FloatTensor(args.n_attr_templet).fill_(1.0), requires_grad=True)

    def forward(self, input):
        if self.mode == 'fc':
            return self.fc_forward(input)
        elif self.mode == 'encoder':
            return self.encode(input, False)
        elif self.mode == 'plain_cosine':
            spt, qry = input
            return self.plain_cosine(spt, qry)
        elif self.mode == 'LPE':
            spt, qry, label2vec, support_ids, labels_train = input
            H, W = spt.shape[-2], spt.shape[-1]
            features_spt = spt.view(self.args.shot, self.args.way, self.encoder_dim, H, W) #[shot, 5, 640, 5, 5]
            features_qry = qry.view(self.args.query * self.args.way, self.encoder_dim, H, W) #[5*15, 640, 5, 5]

            # sem -> filters
            filter_weights = torch.empty((self.args.n_attr_templet, self.args.way, self.encoder_dim, 1, 1)).cuda() #[n_attr_templet,5,640,1,1]
            for i_templet in range(self.args.n_attr_templet):
                filter_weights[i_templet] = getattr(self, 'sem2attr_{}'.format(i_templet))(label2vec).unsqueeze(-1).unsqueeze(-1)

            # mean_normalize
            features_spt = self.mean_normalize(features_spt, -3) #[shot, 5, 640, 5, 5]
            features_qry = self.mean_normalize(features_qry, -3)
            filter_weights = self.mean_normalize(filter_weights, -3)
            
            # spt -> spt_templets
            features_spt_templets = torch.empty((self.args.shot, self.args.way, self.args.n_attr_templet, self.encoder_dim)).cuda()
            for way in range(self.args.way):
                features_spt_way = features_spt[:,way,:,:,:].squeeze(1) #[shot, 640, 5, 5]
                filter_weights_way = filter_weights[:,way,:,:,:].squeeze(1)
                activation_map = torch.sigmoid(F.conv2d(features_spt_way, filter_weights_way)) #[shot, n_attr_templet, 5, 5]
                activation_map = activation_map / activation_map.sum(dim=[-1,-2], keepdim=True)
                features_spt_templets[:,way] = (features_spt_way.unsqueeze(1) * activation_map.unsqueeze(2)).sum([-1,-2]) #[shot,5,n_attr_templet,640]
            features_spt_templets_norm = self.mean_normalize(features_spt_templets, -1)

            # entropy loss
            templets_simi_matrix = torch.bmm(features_spt_templets_norm.reshape(-1, self.args.n_attr_templet, self.encoder_dim), \
                            features_spt_templets_norm.reshape(-1, self.args.n_attr_templet, self.encoder_dim).transpose(1,2)) #[5,n_attr_templet,n_attr_templet]
            loss_diff = templets_simi_matrix.mean()

            # Transfer from base
            if not self.args.no_transferbase:
                labels_train = F.one_hot(torch.arange(self.args.way).cuda(), num_classes=self.args.way).float()
                weight_base_tmp = self.mean_normalize(self.weight_base, -1)
                weight_transfer_from_base = torch.empty((self.args.shot, self.args.way, self.args.n_attr_templet, self.encoder_dim)).cuda()
                if self.training:
                    combined = torch.cat((self.class_no, support_ids))
                    uniques, counts = combined.unique(return_counts=True)
                    difference = uniques[counts == 1]
                    Kbase_ids = difference
                    weight_base_tmp = torch.einsum('ij,jkl->ikl', F.one_hot(Kbase_ids, num_classes=self.args.num_class).float(), weight_base_tmp)
                    for i in range(self.args.n_attr_templet):
                        features_train_i = features_spt_templets_norm[:,:,i,:].reshape(self.args.shot, self.args.way, self.encoder_dim)
                        weight_base_i = weight_base_tmp[:,i,:].view(1, Kbase_ids.shape[0], self.encoder_dim).repeat(self.args.shot,1,1)
                        weight_transfer_from_base[:,:,i,:] = self.transferbase(i, features_train_i, labels_train.unsqueeze(0).repeat(self.args.shot,1,1), weight_base_i, Kbase_ids.unsqueeze(0).repeat(self.args.shot,1))
                else:
                    for i in range(self.args.n_attr_templet):
                        features_train_i = features_spt_templets_norm[:,:,i,:].reshape(self.args.shot, self.args.way, self.encoder_dim)
                        weight_base_i = weight_base_tmp[:,i,:].view(1, self.args.num_class, self.encoder_dim).repeat(self.args.shot,1,1)
                        weight_transfer_from_base[:,:,i,:] = self.transferbase(i, features_train_i, labels_train.unsqueeze(0).repeat(self.args.shot,1,1), weight_base_i, None)
                if self.args.lambda_sqr:
                    features_spt_templets_norm = (self.lamda_novel**2).unsqueeze(0).unsqueeze(0).unsqueeze(-1) * features_spt_templets_norm + \
                                                 (self.lamda_base**2).unsqueeze(0).unsqueeze(0).unsqueeze(-1) * weight_transfer_from_base
                else:
                    features_spt_templets_norm = (self.lamda_novel).unsqueeze(0).unsqueeze(0).unsqueeze(-1) * features_spt_templets_norm + \
                                                 (self.lamda_base).unsqueeze(0).unsqueeze(0).unsqueeze(-1) * weight_transfer_from_base
            else:
                features_spt_templets_norm = features_spt_templets_norm
            features_spt_templets_norm = features_spt_templets_norm.mean(0, keepdim=True) #[1, 5,n_attr_templet,640]
            features_spt_templets_norm = self.mean_normalize(features_spt_templets_norm, -1)

            # qry -> qry_templets
            features_qry_templets = torch.empty((self.args.way, self.args.query*self.args.way, self.args.n_attr_templet, self.encoder_dim)).cuda()
            for way in range(self.args.way):
                filter_weights_way = filter_weights[:,way,:].squeeze(1)
                activation_map = torch.sigmoid(F.conv2d(features_qry, filter_weights_way)) #[shot, n_attr_templet, 5, 5]
                activation_map = activation_map / activation_map.sum(dim=[-1,-2], keepdim=True)
                features_qry_templets[way] = (features_qry.unsqueeze(1) * activation_map.unsqueeze(2)).sum([-1,-2]) #[75,n_attr_templet,640]
            features_qry_templets_norm = self.mean_normalize(features_qry_templets, -1)

            # similarity
            similarity_matrix = torch.mean(torch.einsum('mikl,ijkl->mjik', features_spt_templets_norm, features_qry_templets_norm), dim=0) # [5*15,5,n_attr_templet]

            # similarity weights
            if self.args.templet_weight == 'sem_generate':
                weights_on_templet = self.sem2templet_weight(label2vec) #[5, n_attr_templet]
                if self.args.lambda_sqr:
                    weights_on_templet = (self.tmpweight_lambda1**2) * torch.sigmoid(self.templet_weight_bias) + (self.tmpweight_lambda2**2) * weights_on_templet
                else:
                    weights_on_templet = (self.tmpweight_lambda1) * torch.sigmoid(self.templet_weight_bias) + (self.tmpweight_lambda2) * weights_on_templet
            elif self.args.templet_weight == 'average':
                weights_on_templet = torch.ones(self.args.way, self.args.n_attr_templet).cuda() / self.args.n_attr_templet
            else:
                raise Exception('how to generate templet_weight????')
            weights_on_templet = weights_on_templet / weights_on_templet.sum(dim=-1, keepdim=True)
            similarity_matrix = (similarity_matrix * weights_on_templet.unsqueeze(0)).sum(-1)

            if self.training:
                return similarity_matrix / self.args.temperature, self.fc(features_qry.mean([-1,-2])), loss_diff
            else:
                return similarity_matrix / self.args.temperature
        elif self.mode == 'fc_sem_filters':
            x, label2vec = input
            H, W = x.shape[-2], x.shape[-1]
            n_base = label2vec.shape[0]
            features_x = x.view(self.args.batch, self.encoder_dim, H, W) #[128, 640, 5, 5]

            if not self.args.no_queryatt:
                # sem -> filters
                filter_weights = torch.empty((self.args.n_attr_templet, n_base, self.encoder_dim, 1, 1)).cuda() #[10,64,640,1,1]
                for i_templet in range(self.args.n_attr_templet):
                    filter_weights[i_templet] = getattr(self, 'sem2attr_{}'.format(i_templet))(label2vec).unsqueeze(-1).unsqueeze(-1)
                filter_weights = filter_weights.view(self.args.n_attr_templet*n_base, self.encoder_dim, 1, 1) #[10*64,640,1,1]

                # mean_normalize
                features_x = self.mean_normalize(features_x, -3)
                filter_weights = self.mean_normalize(filter_weights, -3)

                # x -> x_templets
                activation_x = torch.sigmoid(F.conv2d(features_x, filter_weights)) #[128,10*64,5,5]
                activation_x = activation_x / activation_x.sum(dim=[-1,-2], keepdim=True)
                features_x_templets = (features_x.unsqueeze(1) * activation_x.unsqueeze(2)).sum([-1,-2]) #[128,10*64,640]

                #similarity
                features_x_templets_norm = self.mean_normalize(features_x_templets, -1)
                features_x_templets_norm = features_x_templets_norm.view(self.args.batch, self.args.n_attr_templet, n_base, self.encoder_dim) #[128,10,64,640]
                fc_norm = self.mean_normalize(self.weight_base, -1) # [64,10,640]
                similarity_matrix = torch.einsum('ijkl,kjl->ikj', features_x_templets_norm, fc_norm) #[128, 64, 10]
            else:
                # mean_normalize
                features_x_norm = self.mean_normalize(features_x.mean([-1,-2]), -1)
                #similarity
                fc_norm = self.mean_normalize(self.weight_base, -1) # [64,10,640]
                similarity_matrix = torch.einsum('il,kjl->ikj', features_x_norm, fc_norm) #[128, 64, 10]

            # similarity weights
            if self.args.templet_weight == 'sem_generate':
                weights_on_templet = self.sem2templet_weight(label2vec) #[64, 10]
                if self.args.lambda_sqr:
                    weights_on_templet = (self.tmpweight_lambda1**2) * torch.sigmoid(self.templet_weight_bias) + (self.tmpweight_lambda2**2) * weights_on_templet
                else:
                    weights_on_templet = (self.tmpweight_lambda1) * torch.sigmoid(self.templet_weight_bias) + (self.tmpweight_lambda2) * weights_on_templet
            elif self.args.templet_weight == 'average':
                weights_on_templet = torch.ones(n_base, self.args.n_attr_templet).cuda() / self.args.n_attr_templet
            else:
                raise Exception('how to generate templet_weight????')

            weights_on_templet = weights_on_templet / weights_on_templet.sum(dim=-1, keepdim=True)
            similarity_matrix = (similarity_matrix * weights_on_templet.unsqueeze(0)).sum(-1)

            return similarity_matrix / self.args.temperature
        else:
            raise ValueError('Unknown mode')

    def fc_forward(self, x):
        x = x.mean(dim=[-1, -2])
        return self.fc(x)

    def plain_cosine(self, spt, qry):
        spt = spt.squeeze(0)

        # shifting channel activations by the channel mean
        spt_norm = self.normalize_feature(spt)
        qry_norm = self.normalize_feature(qry)

        spt_norm = spt_norm.view(-1, self.args.way, *spt.shape[1:]) # self.args.shot -> -1
        spt_norm = spt_norm.mean(dim=0)

        spt_norm_pooled = spt_norm.mean(dim=[-1, -2])
        qry_norm_pooled = qry_norm.mean(dim=[-1, -2])
        qry_pooled = qry.mean(dim=[-1, -2]) 

        spt_pooled_cos = spt_norm_pooled.unsqueeze(0).repeat(self.args.way*self.args.query,1,1)
        qry_pooled_cos = qry_norm_pooled.unsqueeze(1).repeat(1,self.args.way,1)

        similarity_matrix = F.cosine_similarity(spt_pooled_cos, qry_pooled_cos, dim=-1)

        if self.training:
            return similarity_matrix / self.args.temperature, self.fc(qry_pooled)
        else:
            return similarity_matrix / self.args.temperature


    def mean_normalize(self, x, dim, eps=1e-05):
        x = x - torch.mean(x, dim=dim, keepdim=True)
        x = x / (x.norm(dim=dim, keepdim=True) + eps)
        return x

    def normalize_feature(self, x):
        return x - x.mean(1).unsqueeze(1)

    def encode(self, x, do_gap=True):
        x = self.encoder(x)

        if do_gap:
            return F.adaptive_avg_pool2d(x, 1)
        else:
            return x

    # Adapted from AttentionBasedBlock of Gidaris & Komodakis, CVPR 2018: 
    # https://github.com/gidariss/FewShotWithoutForgetting/architectures/ClassifierWithFewShotGenerationModule.py
    def transferbase(self, n_attr, features_train, labels_train, weight_base, Kbase):
        batch_size, num_train_examples, num_features = features_train.size()
        nKbase = weight_base.size(1) # [batch_size x nKbase x num_features]
        labels_train_transposed = labels_train.transpose(1,2)
        nKnovel = labels_train_transposed.size(1) # [batch_size x nKnovel x num_train_examples]

        features_train = features_train.view(
            batch_size*num_train_examples, num_features)
        Qe = getattr(self,'queryLayer_{}'.format(n_attr))(features_train)
        Qe = Qe.view(batch_size, num_train_examples, self.encoder_dim)
        Qe = F.normalize(Qe, p=2, dim=Qe.dim()-1, eps=1e-12)

        if Kbase is not None:
            wkeys = getattr(self,'wkeys_{}'.format(n_attr))[Kbase.view(-1)] # the keys of the base categoreis
        else:
            wkeys = getattr(self,'wkeys_{}'.format(n_attr)).repeat(batch_size,1,1)
        wkeys = F.normalize(wkeys, p=2, dim=wkeys.dim()-1, eps=1e-12)
        # Transpose from [batch_size x nKbase x nFeat] to
        # [batch_size x self.encoder_dim x nKbase]
        wkeys = wkeys.view(batch_size, nKbase, self.encoder_dim).transpose(1,2)

        # Compute the attention coeficients
        # batch matrix multiplications: AttentionCoeficients = Qe * wkeys ==>
        # [batch_size x num_train_examples x nKbase] =
        #   [batch_size x num_train_examples x nFeat] * [batch_size x nFeat x nKbase]
        AttentionCoeficients = getattr(self,'scale_att_{}'.format(n_attr)) * torch.bmm(Qe, wkeys)
        AttentionCoeficients = F.softmax(
            AttentionCoeficients.view(batch_size*num_train_examples, nKbase), dim=1)
        AttentionCoeficients = AttentionCoeficients.view(
            batch_size, num_train_examples, nKbase)

        # batch matrix multiplications: weight_novel = AttentionCoeficients * weight_base ==>
        # [batch_size x num_train_examples x num_features] =
        #   [batch_size x num_train_examples x nKbase] * [batch_size x nKbase x num_features]
        weight_novel = torch.bmm(AttentionCoeficients, weight_base)
        # batch matrix multiplications: weight_novel = labels_train_transposed * weight_novel ==>
        # [batch_size x nKnovel x num_features] =
        #   [batch_size x nKnovel x num_train_examples] * [batch_size x num_train_examples x num_features]
        weight_novel = torch.bmm(labels_train_transposed, weight_novel)
        weight_novel = weight_novel.div(
            labels_train_transposed.sum(dim=2, keepdim=True).expand_as(weight_novel))

        return weight_novel

