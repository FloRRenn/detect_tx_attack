import numpy as np 
import pandas as pd
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from node2vec import Node2Vec
import json
import operator
import random
from tqdm import tqdm
from ast import literal_eval
import gc

import sys
sys.path.append('./dapps/s2v_lib')
from embedding import EmbedMeanField
from mlp import LSTMTagger

seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# Prepare dataset
df_train = pd.read_csv("./final_dataset/binary/binary_train.csv")
df_test = pd.read_csv('./final_dataset/binary/binary_test.csv')
print("==> Train shape:", df_train.shape)
print("==> Test shape:", df_test.shape)

df_train['senders'] = df_train['senders'].apply(literal_eval)
df_train['to_contracts'] = df_train['to_contracts'].apply(literal_eval)
df_train['tx_list'] = df_train['tx_list'].apply(literal_eval)

df_test['senders'] = df_test['senders'].apply(literal_eval)
df_test['to_contracts'] = df_test['to_contracts'].apply(literal_eval)
df_test['tx_list'] = df_test['tx_list'].apply(literal_eval)

trace_df = pd.read_csv("./data/tx_trace.csv")
trace_df.drop_duplicates(inplace = True)
trace_df["trace_graph"] = trace_df["trace_graph"].apply(json.loads)

def get_trace(tx):
    row = trace_df.loc[trace_df['tx_hash'] == tx]
    if row.empty or row['trace_count'].values[0] == 0:
        return None

    data = row['trace_graph'].values
    return data[0]

class GraphObj(object):
    def __init__(self, graph, gid, node_tags, edge_tags, label, user_node_id, dapp_node_id):
        self.g = graph
        self.gid = gid  # Graph ID
        self.node_tags = node_tags
        self.edge_tags = edge_tags
        self.num_nodes = len(node_tags)

        self.user_node_id = user_node_id  # ID of the user node
        self.dapp_node_id = dapp_node_id  # ID of the dapp node

        self.label = label  # Graph label

        x, y = zip(*graph.edges())
        self.num_edges = len(x)
        self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype = np.int32)  # Array to store edge pairs
        self.edge_pairs[:, 0] = x
        self.edge_pairs[:, 1] = y
        self.edge_pairs = self.edge_pairs.flatten()  # Flatten edge pairs array


def sampling(group, amount, random_state):
    return group.sample(n = amount, frac = None, random_state = random_state, replace = True)

df_train = df_train.groupby('label', group_keys = False).apply(sampling, 3000, seed)
print("==> Train shape after sampling:", df_train.shape)

feat_dict = {}
edge_feat_dict = {}
def make_graph_trace(df : pd.DataFrame, process_name):
    global feat_dict, edge_feat_dict
    graph_seqs = []
    
    # for index, row in df.iterrows():
    for index, row in tqdm(df.iterrows(), total = len(df), desc = "Create sequence graphs for " + process_name):
        graph_lists = []
        label = row['label']
        
        for sender, tx_hash, contract in zip(row['senders'], row['tx_list'], row['to_contracts']):
            trace = get_trace(tx_hash)
            if not trace:
                # print("At label", label, " -- ", tx_hash)
                # print("*" * 40)
                continue
            
            node_dict = {}
            graph = nx.DiGraph()
            graph.label = label
            graph.tx = tx_hash
            
            traces_json = sorted(trace, key = operator.itemgetter(2, 5, 6)) 
            for trace in traces_json:
                if trace[2] == "reference":
                    continue
                
                if not trace[3] in node_dict: # sender address
                    mapped = len(node_dict)
                    node_dict[trace[3]] = mapped
                    
                if not trace[4] in node_dict: # reciever address
                    mapped = len(node_dict)
                    node_dict[trace[4]] = mapped
                    
                node_type_1 = trace[0]['group'] if (trace[0]['label'].startswith('address') or trace[0]['label'].startswith('0x')) else trace[0]['label']
                if not node_type_1 in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[node_type_1] = mapped
                
                node_type_2 = trace[1]['group'] if (trace[1]['label'].startswith('address') or trace[1]['label'].startswith('0x')) else trace[1]['label']
                if not node_type_2 in feat_dict:
                    mapped = len(feat_dict)
                    feat_dict[node_type_2] = mapped
                
                graph.add_node(node_dict[trace[3]], nodetype = node_type_1) # sender node
                graph.add_node(node_dict[trace[4]], nodetype = node_type_2) # reciever node
                
                if trace[2] == 'transfer_to':
                    edge_type = len(str(trace[6]))
                else:
                    edge_type = trace[2]
                if not edge_type in edge_feat_dict:
                    mapped = len(edge_feat_dict)
                    edge_feat_dict[edge_type] = mapped

                graph.add_edge(node_dict[trace[3]], node_dict[trace[4]], edgetype = edge_type) # method was used in tx
            
            node_tags = []
            edge_tags = []      
            for (n, nt) in sorted(graph.nodes(data=True)):
                node_tags.append(feat_dict[nt['nodetype']])

            sns, dns, attrs = zip(*graph.edges(data=True))
            for attr in attrs:
                edge_tags.append(edge_feat_dict[attr['edgetype']])
                
            graph_lists.append(GraphObj(graph, graph.tx, node_tags, edge_tags, label, 
                                            node_dict[sender] if sender in node_dict.keys() else -1, 
                                            node_dict[contract] if (contract != '' and contract != ' ') and contract in node_dict.keys() else -1))
        
        if len(graph_lists) != 0:
            graph_seqs.append(graph_lists)
            
    return graph_seqs

# Create graph sequences from dataset
train_graphs = make_graph_trace(df_train, 'train')
test_graphs = make_graph_trace(df_test, 'test')

del df_train, df_test, trace_df
gc.collect()

class SeqClassifier(nn.Module):
    def __init__(self, output_dim, latent_dim, feat_dim, edge_feat_dim, 
                 max_lv, hidden_size, num_class):
        super(SeqClassifier, self).__init__()
        
        self.feat_dim = feat_dim
        self.edge_feat_dim = edge_feat_dim
        
        if output_dim == 0:
            self.output_dim = latent_dim

        # Embedding layers
        self.s2v = EmbedMeanField(latent_dim = latent_dim, 
                                  output_dim = 0,  # output_dim = out_dim
                                  num_node_feats = feat_dim, 
                                  num_edge_feats = edge_feat_dim,
                                  max_lv = max_lv) 
        
        # LSTM layer
        self.lstm = LSTMTagger(embedding_dim = output_dim, 
                              hidden_dim = hidden_size, 
                              vocab_size = 128*20, 
                              target_size = num_class)
        
        self.my_device = "cuda"
        
    def prepare_feature_label(self, batch_graph):
        # Prepare node, edge features, and labels from a batch of graphs
        labels = torch.LongTensor(len(batch_graph))
        n_nodes = 0
        n_edges = 0
        concat_feat = []
        concat_edge_feat = []
        
        for i, graph in enumerate(batch_graph):
            n_nodes += graph.num_nodes
            n_edges += graph.num_edges
            
            if graph.node_tags is not None:
                concat_feat += graph.node_tags
                
            if graph.edge_tags is not None:
                concat_edge_feat += graph.edge_tags
                concat_edge_feat += graph.edge_tags

        node_feat = None
        if len(concat_feat) != 0:
            concat_feat = torch.LongTensor(concat_feat).view(-1, 1)
            node_feat = torch.zeros(n_nodes, self.feat_dim)
            node_feat.scatter_(1, concat_feat, 1)

        edge_feat = None
        if len(concat_edge_feat) != 0:
            concat_edge_feat = torch.LongTensor(concat_edge_feat).view(-1, 1)
            edge_feat = torch.zeros(n_edges * 2, self.edge_feat_dim)
            edge_feat.scatter_(1, concat_edge_feat, 1)
            
            node_feat = node_feat.cuda() 
            edge_feat = edge_feat.cuda() 
            labels = labels.cuda()

        return node_feat, edge_feat, labels

    def forward(self, batch_graph):
        embed_list = []
        embed_node_eoa_list = []
        embed_node_d_list = []
        labels = torch.LongTensor(len(batch_graph), 1) 
        
        batch_size = len(batch_graph)
        embed_length = []
        max_length = max(len(graph_seq) for graph_seq in batch_graph)

        i = 0
        for graph_seq in batch_graph:
            if len(graph_seq) <= 0:
                continue
            
            embed_temp = None
            embed_node_eoa_temp = None
            embed_node_dapp_temp = None
            
            # Set labels and process each graph sequence
            for g in graph_seq:
                labels[i] = graph_seq[0].label

                # Node2Vec embedding
                node2vec = Node2Vec(g.g, dimensions=64, walk_length=10, num_walks=10, workers=4, quiet=True)
                node2vec_model = node2vec.fit(window=5, min_count=1, batch_words=4)
                
                # Embedding for user and dapp nodes
                ae_user = np.zeros(shape = (64))
                if g.user_node_id != -1:
                    ae_user = node2vec_model.wv[str(g.user_node_id)]

                ae_dapp = np.zeros(shape = (64))
                if g.dapp_node_id != -1:
                    ae_dapp = node2vec_model.wv[str(g.dapp_node_id)]

                embed_node_eoa = torch.FloatTensor(ae_user).unsqueeze(0)
                embed_node_dapp = torch.FloatTensor(ae_dapp).unsqueeze(0)

                embed_node_eoa_temp = torch.cat((embed_node_eoa_temp, embed_node_eoa), 0) if embed_node_eoa_temp is not None else embed_node_eoa
                embed_node_dapp_temp = torch.cat((embed_node_dapp_temp, embed_node_dapp), 0) if embed_node_dapp_temp is not None else embed_node_dapp
                
                # Structural embedding using s2v model
                node_feat, edge_feat, _ = self.prepare_feature_label([g])
                embed_struc = self.s2v([g], node_feat, edge_feat)
                embed_temp = torch.cat((embed_temp, embed_struc), 0) if embed_temp is not None else embed_struc
            
            # Padding for sequences of different lengths
            embed_node_eoa_temp = np.lib.pad(embed_node_eoa_temp.detach().numpy(), pad_width=((0, max_length - len(embed_temp)), (0, 0)),
                                             mode='constant', constant_values=((0, -1), (-1, -1)))
            embed_node_eoa_list.append(embed_node_eoa_temp.tolist())

            embed_node_dapp_temp = np.lib.pad(embed_node_dapp_temp.detach().numpy(), pad_width=((0, max_length-len(embed_temp)), (0, 0)), 
                                              mode='constant', constant_values=((-1, -1), (-1, -1)))
            embed_node_d_list.append(embed_node_dapp_temp.tolist())
            
            embed_length.append(len(embed_temp))
            embed_temp = np.lib.pad(embed_temp.detach().cpu().numpy(), pad_width=((0, max_length-len(embed_temp)), (0, 0)), 
                                    mode='constant', constant_values=((-1, -1), (-1, -1)))
            
            embed_list.append(embed_temp.tolist())

        # Convert embeddings to PyTorch tensors and send them to device
        embed_list = torch.FloatTensor(embed_list).to(device=self.my_device)
        embed_node_eoa_list = torch.FloatTensor(embed_node_eoa_list).to(device=self.my_device)
        embed_node_d_list = torch.FloatTensor(embed_node_d_list).to(device=self.my_device)
        
        return self.lstm(np.array(embed_length), batch_size, embed_list, labels, embed_node_eoa_list, embed_node_d_list)


print("Feat dim:", len(feat_dict))
print("Edge feat dim:", len(edge_feat_dict))

EPOCHS = 30
BATCH_SIZE = 128
NUM_CLASS_BINARY = 2  # for binary classification: attack or non-attack sequence
NUM_CLASS_MULTICLASS = 5  # for multiclass classification

# Init SeqClassifier
classifier = SeqClassifier(out_dim=128, latent_dim=128, feat_dim=len(feat_dict),
                           edge_feat_dim=len(edge_feat_dict), max_lv=4, hidden_size=128, num_class=NUM_CLASS_MULTICLASS)
classifier = classifier.cuda()

opt = optim.Adam(classifier.parameters(), lr=0.0001)

def run_model(epoch, process_name, seq_graphs, model, seq_graph_idx, optimizer=None):
    metrics_scores = []
    
    # Calculate the total number of iterations
    total_iters = (len(seq_graph_idx) + (BATCH_SIZE - 1) * (optimizer is None)) // BATCH_SIZE
    pbar = tqdm(range(total_iters), unit='batch')
    
    n_samples = 0
    print("- Process:", process_name)
    
    for pos in pbar:
        selected_idx = seq_graph_idx[pos * BATCH_SIZE: (pos + 1) * BATCH_SIZE]
        batch_graph = [seq_graphs[idx] for idx in selected_idx]
        logits, loss, acc = model(batch_graph)
        
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
        
        n_samples += len(selected_idx)
        
        # Extract loss and accuracy
        loss = loss.data.cpu().item() 
        acc = acc.data.cpu().item()
        pbar.set_description('\t loss: %0.5f acc: %0.5f' % (loss, acc))
        metrics_scores.append(np.array([loss, acc]) * len(selected_idx))
    
    metrics_scores = np.array(metrics_scores)
    avg_scores = np.sum(metrics_scores, 0) / n_samples
    return avg_scores

# Prepare indices for training and testing
train_idxes = list(range(len(train_graphs)))
test_idxes = list(range(len(test_graphs)))

# Lists to store training and validation scores
train_scores = []
val_scores = []

# Variable to keep track of the best validation loss
best_loss = None

for epoch in range(EPOCHS):
    print("Epoch", epoch)
    
    # Shuffle training indices for each epoch
    random.shuffle(train_idxes)
    
    # Training phase
    train_score = run_model(epoch, "Train", train_graphs, classifier, train_idxes, opt)
    train_scores.append((epoch, train_score[0], train_score[1]))
    print('--> loss %.5f acc %.5f' % (train_score[0], train_score[1]))
    print()
    
    # Validation phase
    val_score = run_model(epoch, "Valid", test_graphs, classifier, test_idxes)
    val_scores.append((epoch, val_score[0], val_score[1]))
    print('--> val_loss %.5f val_acc %.5f' % (val_score[0], val_score[1]))
    
    # Save the model with the best validation loss
    if best_loss is None or val_score[0] < best_loss:
        best_loss = val_score[0]
        print("==> Save this model with the best loss:", best_loss)
        torch.save(classifier.state_dict(), f'./checkpoint/best_loss_model.model')
    print("*" * 50)

scores = {
    'train' : train_scores,
    'val' : val_scores
}
with open("score.json", "w+") as f:
    json.dump(scores, f)