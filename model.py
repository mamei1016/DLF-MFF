from argparse import Namespace
import torch
import torch.nn as nn
from rdkit.Chem import AllChem
from graph import get_two_graph, get_three_graph
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F
from rdkit import Chem
import numpy as np
from rdkit.Chem import Draw
import torchvision.transforms as transforms
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
num_atom_type = 120 #including the extra mask tokens
num_chirality_tag = 3

# ECFP指纹特征学习模型
class FPN(nn.Module):
    def __init__(self,args):
        super(FPN, self).__init__()
        self.fp_dim = args.fp_dim
        self.fp_2_dim=args.fp_2_dim
        self.dropout_fpn = args.dropout_fpn
        self.hidden_dim = args.hidden_size
        self.cuda = args.cuda
        self.args = args

        self.fc1=nn.Linear(self.fp_dim, self.fp_2_dim)
        self.act_func = nn.ReLU()
        self.fc2 = nn.Linear(self.fp_2_dim, self.hidden_dim)
        self.dropout = nn.Dropout(p=self.dropout_fpn)
    
    def forward(self, smile):
        fp_list=[]
        for i, one in enumerate(smile):
            fp=[]
            RDLogger.DisableLog('rdApp.*')
            mol = Chem.MolFromSmiles(one)
            fp_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            fp.extend(fp_morgan)
            fp_list.append(fp)
        fp_list = torch.Tensor(fp_list)
        if self.cuda:
            fp_list = fp_list.cuda()

        fpn_out = self.fc1(fp_list)
        fpn_out = self.dropout(fpn_out)
        fpn_out = self.act_func(fpn_out)
        fpn_out = self.fc2(fpn_out)
        #print('fpn_out', fpn_out.size())
        return fpn_out

# 2D分子图特征学习
class GCN(torch.nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        #torch.manual_seed(12345)
        self.emb_dim= args.emb_dim_gnn
        self.hidden_gnn = args.hidden_gnn
        self.dropout_gnn = args.dropout_gnn
        self.args = args
        self.cuda = args.cuda

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, self.emb_dim)
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, self.emb_dim)
        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)
        self.conv1 = GCNConv(self.emb_dim, self.hidden_gnn)
        self.conv2 = GCNConv(self.hidden_gnn, self.hidden_gnn)
        self.conv3 = GCNConv(self.hidden_gnn, self.hidden_gnn)

        self.l1 = nn.Linear(self.emb_dim,self.emb_dim)
        self.dropout = nn.Dropout(p=self.dropout_gnn)
        self.AN1 = torch.nn.BatchNorm1d(self.emb_dim)
        self.l2 = nn.Linear(self.emb_dim, self.hidden_gnn)
        self.dropout = nn.Dropout(p=self.dropout_gnn)
        self.AN2 = torch.nn.BatchNorm1d(self.hidden_gnn)

    def forward(self,smiles):
        gcn_outs = []
        for i, one in enumerate(smiles):
            #print('gnn_i', i, 'smile', one)
            RDLogger.DisableLog('rdApp.*')
            mol = Chem.MolFromSmiles(one)
            x, atom_size, edge_index = get_two_graph(mol,self.args)
            if self.cuda:
                x, edge_index= x.cuda(), edge_index.cuda()


            x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

            x = self.dropout(F.relu(self.l1(x)))

            # 1. Obtain node embeddings
            h_1 = F.relu(self.conv1(x, edge_index))

            x = self.l2(x)
            # print('x_1gcn', x)

            h_1 = self.dropout(h_1+x)

            h_2 = F.relu(self.conv2(h_1, edge_index))
            #print('x_2gcn', x)
            #x = F.relu(self.conv3(x, edge_index))
            h_2 =self.dropout(h_2+x)



            gcn_out = h_2.sum(dim=0) / atom_size

            gcn_outs.append(gcn_out)

        gcn_outs = torch.stack(gcn_outs, dim=0)
        #print('x_gcn', x.size())
        #graph_gnn = global_mean_pool(x, self.batch)  # [batch_size, hidden_channels]
        #print('gnn_g',  gcn_outs .size())
        # g = self.lin(g)

        return gcn_outs


#3D 分子图特征学习
def index_sum(agg_size, source, idx,cuda):
    """
        source is N x hid_dim [float]
        idx    is N           [int]

        Sums the rows source[.] with the same idx[.];
    """
    tmp = torch.zeros((agg_size, source.shape[1]))
    tmp = tmp.cuda() if cuda else tmp
    res = torch.index_add(tmp, 0, idx, source)
    return res

class ConvEGNN(nn.Module):
    def __init__(self,hid_dim,cuda=True):
        super().__init__()
        self.hid_dim = hid_dim
        self.cuda = cuda
        # computes messages based on hidden representations -> [0, 1]
        self.f_e = nn.Sequential(
            nn.Linear(self.hid_dim * 2 + 1, self.hid_dim), nn.SiLU(),
            nn.Linear(self.hid_dim, self.hid_dim), nn.SiLU())

        # preducts "soft" edges based on messages
        self.f_inf = nn.Sequential(
            nn.Linear(self.hid_dim, 1),
            nn.Sigmoid())

        # updates hidden representations -> [0, 1]
        self.f_h = nn.Sequential(
            nn.Linear(self.hid_dim + self.hid_dim, self.hid_dim), nn.SiLU(),
            nn.Linear(self.hid_dim, self.hid_dim))

        self.f_pos = nn.Sequential(
            nn.Linear(self.hid_dim * 2 + 1, 3), nn.SiLU(),
            nn.Linear(3, 3), nn.SiLU())
        # preducts "soft" edges based on messages
        self.f_h_pos = nn.Sequential(
            nn.Linear(6, 3), nn.SiLU(),
            nn.Linear(3, 3))

    def forward(self, x, edge_index, pos):
        e_st, e_end = edge_index[:, 0], edge_index[:, 1]
        dists = torch.norm(pos[e_st] - pos[e_end], dim=1).reshape(-1, 1)

        # compute messages
        tmp = torch.hstack([x[e_st], x[e_end], dists])
        m_ij = self.f_e(tmp)
        m_ij_pos = self.f_pos(tmp)

        # predict edges
        e_ij = self.f_inf(m_ij)

        # average e_ij-weighted messages
        # m_i is num_nodes x hid_dim
        # m_i_pos is num_nodes x 3  (zuobiao)
        m_i = index_sum(x.shape[0], e_ij * m_ij,edge_index[:, 0],self.cuda)
        m_i_pos = 1 / pos.size(0) * index_sum(pos.shape[0], dists * m_ij_pos, edge_index[:, 0],self.cuda)

        # update hidden representations
        x += self.f_h(torch.hstack([x, m_i]))
        pos += self.f_h_pos(torch.hstack([pos, m_i_pos]))

        return  x,  edge_index, pos


class NetEGNN(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.emb_dim = args.emb_egnn
        self.batch = args.batch_size
        self.pool = global_mean_pool
        self.dropout_egnn = args.dropout_egnn
        self.cuda = args.cuda

        self.x_embedding1 = torch.nn.Embedding(num_atom_type, self.emb_dim )
        self.x_embedding2 = torch.nn.Embedding(num_chirality_tag, self.emb_dim )

        torch.nn.init.xavier_uniform_(self.x_embedding1.weight.data)
        torch.nn.init.xavier_uniform_(self.x_embedding2.weight.data)

        self.conv1 = ConvEGNN(self.emb_dim,self.cuda)
        self.conv2 =ConvEGNN(self.emb_dim,self.cuda)
        self.conv3 = ConvEGNN(self.emb_dim,self.cuda )
        self.l1 = nn.Linear(self.emb_dim, self.emb_dim)
        self.AN1 = torch.nn.BatchNorm1d(self.emb_dim)
        self.dropout = nn.Dropout(p=self.dropout_egnn)

    def forward(self, smiles):
        egnn_outs = []
        for i, one in enumerate(smiles):
            #print('i', i,'smile',one)
            RDLogger.DisableLog('rdApp.*')
            mol = Chem.MolFromSmiles(one)

            x, atom_size, edge_index, pos = get_three_graph(mol,self.args)
            #print('i_edge_index', i,edge_index)
            if self.cuda:
                x, edge_index, pos = x.cuda(), edge_index.cuda(), pos.cuda()

            x = self.x_embedding1(x[:, 0]) + self.x_embedding2(x[:, 1])

            x = self.dropout(F.relu(self.l1(x)))
            #h = self.emb(h)
            h_1, edge_index, pos = self.conv1(x, edge_index,pos)
            #print('x_1',x)

            h_1 = self.dropout(h_1+x)

            h_2, edge_index, pos = self.conv2(h_1, edge_index,pos)
            #print('x_2', x)
            #x, edge_index, pos = self.conv3(x, edge_index,pos)
            h_2 = self.dropout(h_2+x)

            egnn_out = h_2.sum(dim=0) / atom_size
            #print('one_egnn_out', egnn_out.size())
            egnn_outs.append(egnn_out)

        egnn_outs = torch.stack(egnn_outs, dim=0)

        #print('egnn_g', egnn_outs.size())

        return egnn_outs

# 图像特征学习模型
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.RandomResizedCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

#准备测试所用的模型
class VggNet(nn.Module):
    def __init__(self, args):
        super(VggNet, self).__init__()
        self.image_dim = args.linear_dim
        self.cuda = args.cuda
        self.Conv = torch.nn.Sequential(
            # 3*224*224  conv1
            torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # 64*112*112   conv2
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # 128*56*56    conv3
            torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            # 256*28*28    conv4
            torch.nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
        # 512*14*14   conv5
            torch.nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(512, 512, kernel_size = 3, stride = 1, padding = 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size = 2, stride = 2))
        # 512*7*7
        self.Classes = torch.nn.Sequential(
            torch.nn.Linear(512*7*7, 4096),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.Linear(4096, 1060),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            nn.Linear(1060, self.image_dim))

    def forward(self, smile):
        img_list = []
        for i, one in enumerate(smile):
            img_f = []
            RDLogger.DisableLog('rdApp.*')
            mol = Chem.MolFromSmiles(one)
            img = Draw.MolsToGridImage([mol], molsPerRow=1, subImgSize=(224, 224))
            img = np.array(img)
            img = transform(img)
            # img = img.unsqueeze(0)
            # print(img.size())
            # print('img',img)
            img_list.append(img)
        # print('img_list', img_list.shape)
        img_list = torch.stack(img_list)
        # print('img_list', img_list.size())
        if self.cuda:
            img_list = img_list.cuda()

        x = self.Conv(img_list)
        #print('x_conv', x.size())  # [batchsize,2048]
        fc_input = x.view(x.size(0), -1)
        #print('fc_input', fc_input.size())  # [batchsize,2048]
        #x = x.view(-1, 14 * 14 * 512)
        x = self.Classes(fc_input)
        return x

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, input_dim,n_head,ouput_dim ):

        super(MultiHeadAttention, self).__init__()
        self.input_dim = input_dim
        self.n_heads = n_head
        self.ouput_dim = ouput_dim
        self.d_k = self.d_v = self.input_dim // self.n_heads
        self.W_Q = torch.nn.Linear(self.input_dim, self.d_k * self.n_heads, bias=False)
        self.W_K = torch.nn.Linear(self.input_dim, self.d_k * self.n_heads, bias=False)
        self.W_V = torch.nn.Linear(self.input_dim, self.d_v * self.n_heads, bias=False)
        self.fc = torch.nn.Linear(self.n_heads * self.d_v, self.input_dim, bias=False)
        #self.AN1 = torch.nn.LayerNorm(self.input_dim )
        self.l1 = torch.nn.Linear(self.input_dim , self.ouput_dim)

    def forward(self, X):
        ## (S, D) -proj-> (S, D_new) -split-> (S, H, W) -trans-> (H, S, W)
        Q = self.W_Q(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        K = self.W_K(X).view(-1, self.n_heads, self.d_k).transpose(0, 1)
        V = self.W_V(X).view(-1, self.n_heads, self.d_v).transpose(0, 1)

        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        # context: [n_heads, len_q, d_v], attn: [n_heads, len_q, len_k]
        attn = torch.nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        # context: [len_q, n_heads * d_v]
        context = context.transpose(1, 2).reshape(-1, self.n_heads * self.d_v)
        output = self.fc(context)
        #output = self.AN1(output)
        output =self.l1(output )
        return output
# 定义最终模型
feature_out = []


class FpgnnModel(nn.Module):
    def __init__(self, is_classif,cuda, dropout_fpn,):
        super(FpgnnModel, self).__init__()
        self.is_classif = is_classif
        self.dropout_fpn = dropout_fpn
        self.cuda=cuda
        if self.is_classif:
            self.sigmoid = nn.Sigmoid()

    def create_fpn(self, args):
        self.encoder1 = FPN(args)

    def create_gnn(self, args):
        self.encoder2 = GCN(args)

    def create_egnn(self, args):
        self.encoder3 = NetEGNN(args)

    def create_imagecnn(self, args):
        self.encoder4 = VggNet(args)

    def create_fc(self, args):
        fpn_dim = args.hidden_size
        gnn_dim = args.hidden_gnn
        egnn_dim = args.emb_egnn
        linear_dim = int(args.linear_dim)
        encoder1_FC =nn.Sequential()
        encoder1_FC.add_module('fc',  nn.Linear(fpn_dim, linear_dim)) #2048=128*4*4,此处维数维初始维数*最后一层输出维数
        #encoder1_FC.add_module('LayerNorm_', nn.BatchNorm1d(linear_dim))
        self.encoder1_FC = encoder1_FC
        encoder2_FC = nn.Sequential()
        encoder2_FC.add_module('fc', nn.Linear(gnn_dim, linear_dim))
        #encoder2_FC.add_module('LayerNorm', nn.BatchNorm1d(linear_dim))
        self.encoder2_FC = encoder2_FC
        encoder3_FC = nn.Sequential()
        encoder3_FC.add_module('fc', nn.Linear(egnn_dim, linear_dim))
        #encoder3_FC.add_module('LayerNorm_', nn.BatchNorm1d(linear_dim))
        self.encoder3_FC = encoder3_FC
        encoder4_FC = nn.Sequential()
        encoder4_FC.add_module('fc', nn.Linear(linear_dim, linear_dim))
        #encoder4_FC.add_module('LayerNorm_', nn.BatchNorm1d(linear_dim))
        self.encoder4_FC = encoder4_FC
        self.act_func = nn.ReLU()
        self.fpn_attn = MultiHeadAttention(fpn_dim, 1, linear_dim)
        self.gnn_attn = MultiHeadAttention(gnn_dim, 1, linear_dim)
        self.egnn_attn = MultiHeadAttention(egnn_dim, 1, linear_dim)
        self.vggnet_attn = MultiHeadAttention(linear_dim, 1, linear_dim)


    def create_ffn(self, args):
        linear_dim = int(args.linear_dim)
        self.ffn = nn.Sequential(
        nn.Linear(in_features=linear_dim , out_features=linear_dim, bias=True),
        nn.ReLU(),
        #nn.BatchNorm1d(linear_dim),
        nn.Dropout(self.dropout_fpn),
        nn.Linear(in_features=linear_dim, out_features=args.task_num, bias=True)

        )


    def forward(self, input):
        fpn_out = self.encoder1(input)
        gcn_out = self.encoder2(input)
        egnn_out = self.encoder3(input)
        image_out = self.encoder4(input)
        fpn_out = self.encoder1_FC(fpn_out)
        fpn_out = self.act_func(fpn_out)
        #print('fpn_out:', fpn_out.size())
        gcn_out = self.encoder2_FC(gcn_out)
        gcn_out = self.act_func(gcn_out)
        #print('gnn_out:', gcn_out.size())
        egnn_out = self.encoder3_FC(egnn_out)
        egnn_out = self.act_func(egnn_out)
        #print('egnn_out:', egnn_out.size())
        image_out = self.encoder4_FC(image_out)
        image_out = self.act_func(image_out)
        #print('img_out:', image_out.size())
        #fpn_out = self.fpn_attn(fpn_out)
        #gcn_out = self.gnn_attn(gcn_out)
        #egnn_out = self.egnn_attn(egnn_out)
        #image_out = self.vggnet_attn(image_out)

        #output = torch.cat([fpn_out, gcn_out, egnn_out, image_out], axis=1)

        #output = self.attn(output)
        #print('output:', output.size())
        #output =torch.cat([gcn_out,image_out])
        output = fpn_out+gcn_out+egnn_out+image_out
        output = self.ffn(output)

        #保存特征
        #graph_out = output
        #graph = graph_out.cpu().numpy()
        #graph = graph.tolist()
        #feature_out.append(graph)



        if self.is_classif and not self.training:
            output = self.sigmoid(output)

        return output


def get_features_out():
    return  feature_out


def FPGNN(args):
    if args.dataset_type == 'classification':
        is_classif = 1
    else:
        is_classif = 0
    model = FpgnnModel(is_classif,args.cuda,args.dropout_fpn)
    model.create_fpn(args)
    model.create_gnn(args)
    model.create_egnn(args)
    model.create_imagecnn(args)
    model.create_fc(args)
    model.create_ffn(args)
    #model.create_att(args)

    for param in model.parameters():
        if param.dim() == 1:
            nn.init.constant_(param, 0)
        else:
            nn.init.xavier_normal_(param)

    return model