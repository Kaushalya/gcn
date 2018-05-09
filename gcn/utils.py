import numpy as np
import pickle as pkl
import networkx as nx
import scipy.sparse as sp
from scipy.sparse.linalg.eigen.arpack import eigsh
import sys
from sklearn.preprocessing import OneHotEncoder

def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


def load_directed_data(dataset_str):
    print("loading directed data")
    data = pkl.load(open("data/{}.data".format(dataset_str), 'rb'))
    adj = nx.adjacency_matrix(data['NXGraph'].to_undirected())
    features = data['CSRFeatures'].tolil()
    labels = data['Labels']
    
    encoder = OneHotEncoder(dtype=np.int16)
    labels = encoder.fit_transform(labels.reshape(-1,1))
    labels = labels.toarray()
      
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)
    
    #features[test_idx_reorder, :] = features[test_idx_range, :]
    #labels[test_idx_reorder, :] = labels[test_idx_range, :]
    idx_test = test_idx_range.tolist()
    idx_train = range(140)
    idx_val = range(140, 140+500)
    
    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])
    
    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]
    
    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask
    
def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    ind.dataset_str.x => the feature vectors of the training instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.tx => the feature vectors of the test instances as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.allx => the feature vectors of both labeled and unlabeled training instances
        (a superset of ind.dataset_str.x) as scipy.sparse.csr.csr_matrix object;
    ind.dataset_str.y => the one-hot labels of the labeled training instances as numpy.ndarray object;
    ind.dataset_str.ty => the one-hot labels of the test instances as numpy.ndarray object;
    ind.dataset_str.ally => the labels for instances in ind.dataset_str.allx as numpy.ndarray object;
    ind.dataset_str.graph => a dict in the format {index: [index_of_neighbor_nodes]} as collections.defaultdict
        object;
    ind.dataset_str.test.index => the indices of test instances in graph, for the inductive setting as list object.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    idx_test = test_idx_range.tolist()
    idx_train = range(len(y))
    idx_val = range(len(y), len(y)+500)

    train_mask = sample_mask(idx_train, labels.shape[0])
    val_mask = sample_mask(idx_val, labels.shape[0])
    test_mask = sample_mask(idx_test, labels.shape[0])

    y_train = np.zeros(labels.shape)
    y_val = np.zeros(labels.shape)
    y_test = np.zeros(labels.shape)
    y_train[train_mask, :] = labels[train_mask, :]
    y_val[val_mask, :] = labels[val_mask, :]
    y_test[test_mask, :] = labels[test_mask, :]

    return adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask


def sparse_to_tuple(sparse_mx):
    """Convert sparse matrix to tuple representation."""
    def to_tuple(mx):
        if not sp.isspmatrix_coo(mx):
            mx = mx.tocoo()
        coords = np.vstack((mx.row, mx.col)).transpose()
        values = mx.data
        shape = mx.shape
        return coords, values, shape

    if isinstance(sparse_mx, list):
        for i in range(len(sparse_mx)):
            sparse_mx[i] = to_tuple(sparse_mx[i])
    else:
        sparse_mx = to_tuple(sparse_mx)

    return sparse_mx


def preprocess_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return sparse_to_tuple(features)


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def preprocess_adj(adj, weighted=True):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
   
    adj_normalized = normalize_adj(adj + sp.eye(adj.shape[0]))
    
    if weighted:
        g = nx.from_scipy_sparse_matrix(adj)
        
        print "calculating edge betweenness centrality"
        ebw = nx.edge_betweenness_centrality(g)
        ebw_mat = sp.dok_matrix(adj.shape, dtype=np.float64)
        
        for loc, value in ebw.items():
            ebw_mat[loc] = 1./value
        
        ebw_mat = ebw_mat.tocoo()    
        adj_normalized = np.multiply(adj_normalized, ebw_mat)
                         
    return sparse_to_tuple(sp.coo_matrix(adj_normalized))    
    
def preprocess_mage(adj):
    """Preprocessing of adjacency matrix to motif matrix
    Protype: triangle motif"""
    
    #TODO add other moifs here
    coocurence_count = {}
    wedge_count = {}
    
    g = nx.from_scipy_sparse_matrix(adj)
    for node in g:
        n_set = set(g.neighbors(node))        
        
        for neig in n_set:
            nn_set = set(g.neighbors(neig))
            intersect = n_set.intersection(nn_set)
            diff = n_set.difference(nn_set)
            
            if (node, neig) not in coocurence_count:
                if intersect:
                    coocurence_count[(node, neig)] = len(intersect)+1
                    coocurence_count[(neig, node)] = len(intersect)+1
                else:
                    coocurence_count[(node, neig)] = 1
                    coocurence_count[(neig, node)] = 1
                if diff:
                    wedge_count[(node, neig)] = len(diff)+1
                    wedge_count[(neig, node)] = len(diff)+1
                else:
                    wedge_count[(node, neig)] = 1
                    wedge_count[(neig, node)] = 1
                
    row = np.array([i for i, _ in coocurence_count.keys()])
    col = np.array([j for _, j in coocurence_count.keys()])
    data = np.array([v for v in coocurence_count.values()])
    m_adj = sp.csr_matrix((data, (row, col)), shape=adj.shape)
    
    w_row = np.array([i for i, _ in wedge_count.keys()])
    w_col = np.array([j for _, j in wedge_count.keys()])
    w_data = np.array([v for v in wedge_count.values()])
    w_adj = sp.csr_matrix((w_data, (w_row, w_col)), shape=adj.shape)
    
    motif_mats = list()
    motif_mats.append(preprocess_adj(m_adj))
    motif_mats.append(preprocess_adj(w_adj))

    return motif_mats
    
def preprocess_directed_mage(adj):
    return None
    

def construct_feed_dict(features, support, labels, labels_mask, placeholders, support_wp=None):
    """Construct feed dictionary."""
    feed_dict = dict()
    feed_dict.update({placeholders['labels']: labels})
    feed_dict.update({placeholders['labels_mask']: labels_mask})
    feed_dict.update({placeholders['features']: features})
    feed_dict.update({placeholders['support'][i]: support[i] for i in range(len(support))})
    feed_dict.update({placeholders['num_features_nonzero']: features[1].shape})
    
    if support_wp:
        feed_dict.update({placeholders['wp_support'][i]: support_wp[i] for i in range(len(support_wp))})
    return feed_dict


def chebyshev_polynomials(adj, k):
    """Calculate Chebyshev polynomials up to order k. Return a list of sparse matrices (tuple representation)."""
    print("Calculating Chebyshev polynomials up to order {}...".format(k))

    adj_normalized = normalize_adj(adj)
    laplacian = sp.eye(adj.shape[0]) - adj_normalized
    largest_eigval, _ = eigsh(laplacian, 1, which='LM')
    scaled_laplacian = (2. / largest_eigval[0]) * laplacian - sp.eye(adj.shape[0])

    t_k = list()
    t_k.append(sp.eye(adj.shape[0]))
    t_k.append(scaled_laplacian)

    def chebyshev_recurrence(t_k_minus_one, t_k_minus_two, scaled_lap):
        s_lap = sp.csr_matrix(scaled_lap, copy=True)
        return 2 * s_lap.dot(t_k_minus_one) - t_k_minus_two

    for i in range(2, k+1):
        t_k.append(chebyshev_recurrence(t_k[-1], t_k[-2], scaled_laplacian))

    return sparse_to_tuple(t_k)

