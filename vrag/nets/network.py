import torch
import torch.nn as nn

from vrag.nets import BatchLinear


class Vrag(nn.Module):
    def __init__(self, config):
        super(Vrag, self).__init__()
        self.in_ndims = config['in-ndims']
        self.out_ndims = config['out-ndims']

        gnn_config = config['gnn']
        self.gnn_in_ndims = gnn_config['in-ndims']
        self.gnn_base_ndims = gnn_config['base-ndims']
        self.gnn_out_ndims = gnn_config['out-ndims']

        downsample_config = config['downsample']
        self.downsample_net = BatchLinear(in_ndims=self.in_ndims, out_ndims=self.gnn_in_ndims, bias=True,
                                          config=downsample_config)

        gnn_name = gnn_config['name']
        att_in_ndims = 3
        assert gnn_name == 'base-input-all'
        self.gnn = BatchBaseGnn(config=gnn_config)
        embedding_in_ndims = (self.gnn_base_ndims * 2) + self.gnn_out_ndims + self.gnn_in_ndims + self.in_ndims
        self.attention_net = nn.Linear(in_features=att_in_ndims, out_features=1)  # there are three similarity matrices

        assert gnn_config['activation'] == downsample_config['activation'] == 'elu', 'only works for elu function'
        self.embedding_net = nn.Sequential(
            nn.Linear(in_features=embedding_in_ndims, out_features=self.out_ndims, bias=True),
            nn.ELU(inplace=True),
            nn.Linear(in_features=self.out_ndims, out_features=self.out_ndims, bias=True)
        )

    @staticmethod
    def get_concatenated_layer_features(layer_features: list):
        nlayers = len(layer_features)
        nbatch = len(layer_features[0])

        batch_concatenated_features = []
        for i in range(nbatch):
            batch_features = [layer_features[j][i] for j in range(nlayers)]
            concatenated_feature = torch.cat(batch_features, dim=1)
            batch_concatenated_features.append(concatenated_feature)
        return batch_concatenated_features

    def forward(self, batch_rmac_features: list):
        """
        :param batch_rmac_features: T x D x sqrt(R) x sqrt(R) features
        :return: 1 x out-ndim embedding vector
        """
        # downsample node features for ease of computing
        batch_nframes = [rmac_features.shape[0] for rmac_features in batch_rmac_features]
        _, ndims, h, w = batch_rmac_features[0].shape
        batch_rmac_features = [rmac_features.permute(0, 2, 3, 1) for rmac_features in batch_rmac_features]
        nregions = h * w
        batch_nfeatures = [nframes * nregions for nframes in batch_nframes]

        batch_features = [rmac_features.reshape(batch_nfeatures[i], ndims)
                          for i, rmac_features in enumerate(batch_rmac_features)]
        batch_gnn_input_features = self.downsample_net(batch_features)
        batch_gnn_input_features = [features.view(batch_nframes[i], nregions, self.gnn_in_ndims) for i, features in
                                    enumerate(batch_gnn_input_features)]

        layer_batch_features, layer_batch_similarities = self.gnn(batch_gnn_input_features)
        assert isinstance(layer_batch_features, list)
        layer_batch_features.insert(0, batch_features)  # add the original rmac features
        batch_gnn_features = Vrag.get_concatenated_layer_features(layer_features=layer_batch_features)
        batch_mean_similarities = Vrag.get_concatenated_layer_features(layer_features=layer_batch_similarities)

        # get node weights
        all_similarities = torch.cat(batch_mean_similarities, dim=0)
        all_node_attention = self.attention_net(all_similarities)
        batch_embedding_inputs = []
        start = 0
        for i, nfeatures in enumerate(batch_nfeatures):
            end = start + nfeatures
            node_attention = all_node_attention[start:end]
            gnn_features = batch_gnn_features[i].permute(1, 0)  # (D1 + D2 + D3) x Ni
            start = end

            node_attention = torch.softmax(node_attention, dim=0).view(-1, 1)  # Ni x 1
            embedding_inputs = torch.matmul(gnn_features, node_attention).view(-1)  # (D1 + D2 + D3) embedding
            batch_embedding_inputs.append(embedding_inputs)
        all_embedding_inputs = torch.stack(batch_embedding_inputs, dim=0)  # B x (D1 + D2 + D3) embedding inputs
        all_embedding_outputs = self.embedding_net(all_embedding_inputs)  # B x Do embedding outputs
        return all_embedding_outputs


def _get_base_adjacency(num_frames: int, num_regions: int) -> (torch.Tensor, torch.Tensor):
    """
    Returns every adjacency matrix such that every region is connected to its spatially local neighbors and itself, and
    also to regions in adjacency frames. Regions in the start and end frames will be connected to less neighbors,
    so we will return two adjacency matrices: 2R x 2R start_end_adjacency; (T-2)R x 3R middle_adjacency
    Args:
        num_frames: the number of frames in the video
        num_regions: the number of regions per frame in teh video
    Returns:
    """
    with torch.no_grad():
        frame_adjacency = torch.arange(-num_regions, num_regions * 2).view(1, num_regions * 3)  # 1 x 3R
        frame_adjacency = frame_adjacency.expand(num_regions, num_regions * 3)  # R x 3R

        offsets = (torch.arange(num_frames) * num_regions).view(num_frames, 1, 1)  # T x 1 x 1
        adjacency = frame_adjacency.view(1, num_regions, num_regions * 3)  # 1 x R x 3R
        adjacency = adjacency.expand(num_frames, num_regions, num_regions * 3) + offsets  # T x R x 3R
        mid_adjacency = adjacency[1:-1, :, :]

        start_adjacency = adjacency[0, :, num_regions:]  # R x 3R
        end_adjacency = adjacency[-1, :, :-num_regions]  # R x 3R

        start_end_adjacency = torch.cat([start_adjacency, end_adjacency], dim=0)  # 2R x 2R
        mid_adjacency = mid_adjacency.view(num_regions * (num_frames - 2), num_regions * 3)  # (T-2)R x 3R
    return start_end_adjacency, mid_adjacency


def get_batch_base_adjacencys(batch_num_frames: list, num_regions: int, device) -> (list, list):
    batch_start_end_adjacencys, batch_middle_adjacencys = [], []
    for num_frames in batch_num_frames:
        start_end_adjacency, middle_adjacency = _get_base_adjacency(num_frames=num_frames, num_regions=num_regions)
        start_end_adjacency = start_end_adjacency.to(device=device)
        middle_adjacency = middle_adjacency.to(device=device)
        batch_start_end_adjacencys.append(start_end_adjacency)
        batch_middle_adjacencys.append(middle_adjacency)
    return batch_start_end_adjacencys, batch_middle_adjacencys


class BatchBaseGnn(nn.Module):
    def __init__(self, config):
        super(BatchBaseGnn, self).__init__()
        self.in_ndims = config['in-ndims']
        self.base_ndims = config['base-ndims']
        self.out_ndims = config['out-ndims']

        self.neighbor_aggregation = config['neighbor-aggregation']
        assert self.neighbor_aggregation == 'weighted-sum'

        in_ndims1, out_ndims1 = self.in_ndims, self.base_ndims
        in_ndims2, out_ndims2 = self.base_ndims, self.base_ndims
        in_ndims3, out_ndims3 = self.base_ndims, self.out_ndims

        similarity_config = config['similarity']
        self.batch_similarity_fn1 = get_batch_pairwise_similarity(in_ndims1, config=similarity_config)
        self.batch_similarity_fn2 = get_batch_pairwise_similarity(in_ndims2, config=similarity_config)
        self.batch_similarity_fn3 = get_batch_pairwise_similarity(in_ndims3, config=similarity_config)

        self.gc_layer1 = BatchLinear(in_ndims=in_ndims1, out_ndims=out_ndims1, bias=True, config=config)
        self.gc_layer2 = BatchLinear(in_ndims=in_ndims2, out_ndims=out_ndims2, bias=True, config=config)
        self.gc_layer3 = BatchLinear(in_ndims=in_ndims3, out_ndims=out_ndims3, bias=True, config=config)

    @staticmethod
    def _get_aggregated_neighbors(graph_features: torch.Tensor, similarity_mtx: torch.Tensor,
                                  adjacency: torch.Tensor) -> torch.Tensor:
        """

        Args:
            graph_features:  N x D graph features
            similarity_mtx:  Q x N similarity matrix
            adjacency: Q x K adjacency matrix
        Returns:
        """
        num_queries, num_neighbors = adjacency.shape
        num_features, num_dims = graph_features.shape

        # get neighbor weights
        similarities = similarity_mtx.view(num_queries * num_features)
        offset = (torch.arange(num_queries, dtype=torch.long, device=adjacency.device) * num_features)
        offset = offset.view(num_queries, 1)  # Q x 1
        similarity_idxs = (adjacency + offset).view(num_queries * num_neighbors)  # QK
        neighbor_similarities = similarities[similarity_idxs].view(num_queries, num_neighbors)  # Q x K
        neighbor_weights = torch.softmax(neighbor_similarities, dim=1)  # Q x K
        neighbor_weights = torch.unsqueeze(neighbor_weights, dim=1)  # Q x 1 x K

        neighbor_idxs = adjacency.view(-1)
        neighbor_features = graph_features[neighbor_idxs, :].view(num_queries, num_neighbors, num_dims)  # Q x K x C
        aggregated_features = torch.bmm(neighbor_weights, neighbor_features)  # Q x 1 x C
        aggregated_features = torch.squeeze(aggregated_features, dim=1)  # Q x C
        return aggregated_features

    @staticmethod
    def graph_conv(batch_features: list, batch_start_end_adjacencys: list, batch_middle_adjacencys: list,
                   batch_similarity_mtxs: list, net: BatchLinear, num_regions: int) -> list:
        """
        :param batch_features: [N1 x Di, N2 x Di, ..., Nn x Di] graph node features
        :param batch_similarity_mtxs:  [N1 x N1, N2 x N2, ..., Nn x Nn] graph similarity matrices
        :param batch_start_end_adjacencys:  [[start_end_adj1, mid_adj1], ..., [start_end_adjn, mid_adjn]] graph base adjacency lists
        :param net: graph convolution layer for base adjacency lists
        :param batch_middle_adjacencys:  number aggregator function class
        :param num_regions:  number of regions for each graph
        :return:
        """
        batch_aggregated = []
        num_batches = len(batch_features)
        for i in range(num_batches):
            features = batch_features[i]
            start_end_adjacency = batch_start_end_adjacencys[i]
            middle_adjacency = batch_middle_adjacencys[i]
            similarity_mtx = batch_similarity_mtxs[i]

            start_end_similarity_mtx = torch.cat([similarity_mtx[:num_regions], similarity_mtx[-num_regions:]], dim=0)
            middle_similarity_mtx = similarity_mtx[num_regions:-num_regions]
            start_end_aggregated = BatchBaseGnn._get_aggregated_neighbors(graph_features=features,
                                                                          similarity_mtx=start_end_similarity_mtx,
                                                                          adjacency=start_end_adjacency)
            middle_aggregated = BatchBaseGnn._get_aggregated_neighbors(graph_features=features,
                                                                       similarity_mtx=middle_similarity_mtx,
                                                                       adjacency=middle_adjacency)
            aggregated = torch.cat([start_end_aggregated[:num_regions], middle_aggregated,
                                    start_end_aggregated[-num_regions:]], dim=0)
            batch_aggregated.append(aggregated)
        batch_out_features = net(batch_aggregated)
        return batch_out_features

    def forward(self, batch_in_features: list):
        batch_num_frames = [rmac_features.shape[0] for rmac_features in batch_in_features]
        _, num_regions, ndims = batch_in_features[0].shape
        device = batch_in_features[0].device
        batch_nfeatures = [num_frames * num_regions for num_frames in batch_num_frames]
        batch_features = [input_features.view(batch_nfeatures[i], ndims)
                          for i, input_features in enumerate(batch_in_features)]

        batch_start_end_adjacencys, batch_middle_adjacencys = \
            get_batch_base_adjacencys(batch_num_frames=batch_num_frames, num_regions=num_regions, device=device)

        batch_similarity_mtxs1 = self.batch_similarity_fn1(batch_features)
        batch_features1 = self.graph_conv(batch_features=batch_features,
                                          batch_start_end_adjacencys=batch_start_end_adjacencys,
                                          batch_middle_adjacencys=batch_middle_adjacencys,
                                          batch_similarity_mtxs=batch_similarity_mtxs1, net=self.gc_layer1,
                                          num_regions=num_regions)

        batch_similarity_mtxs2 = self.batch_similarity_fn2(batch_features1)
        batch_features2 = self.graph_conv(batch_features=batch_features1,
                                          batch_start_end_adjacencys=batch_start_end_adjacencys,
                                          batch_middle_adjacencys=batch_middle_adjacencys,
                                          batch_similarity_mtxs=batch_similarity_mtxs2, net=self.gc_layer2,
                                          num_regions=num_regions)

        batch_similarity_mtxs3 = self.batch_similarity_fn3(batch_features2)
        batch_features3 = self.graph_conv(batch_features=batch_features2,
                                          batch_start_end_adjacencys=batch_start_end_adjacencys,
                                          batch_middle_adjacencys=batch_middle_adjacencys,
                                          batch_similarity_mtxs=batch_similarity_mtxs3, net=self.gc_layer3,
                                          num_regions=num_regions)

        batch_mean_similiarities1 = [torch.mean(mtx, dim=1, keepdim=True) for mtx in batch_similarity_mtxs1]
        batch_mean_similiarities2 = [torch.mean(mtx, dim=1, keepdim=True) for mtx in batch_similarity_mtxs2]
        batch_mean_similiarities3 = [torch.mean(mtx, dim=1, keepdim=True) for mtx in batch_similarity_mtxs3]

        batch_features = [batch_features, batch_features1, batch_features2, batch_features3]
        batch_mean_similarities = [batch_mean_similiarities1, batch_mean_similiarities2, batch_mean_similiarities3]
        return batch_features, batch_mean_similarities


def get_batch_pairwise_similarity(in_ndims: int, config: dict):
    similarity = config['name']
    if similarity == 'attention':
        pairwise_similarity = BatchAttentionSimilarity(in_ndims=in_ndims, config=config)
    else:
        raise ValueError('{} is not a valid similarity value'.format(similarity))
    return pairwise_similarity


class BatchAttentionSimilarity(nn.Module):
    def __init__(self, in_ndims, config: dict):
        super(BatchAttentionSimilarity, self).__init__()
        out_ndims = config['out-ndims']
        bias = config['bias']

        self.query_net = nn.Linear(in_features=in_ndims, out_features=out_ndims, bias=bias)
        self.key_net = nn.Linear(in_features=in_ndims, out_features=out_ndims, bias=bias)

    def forward(self, batch_features: list):
        batch_nfeatures = [features.shape[0] for features in batch_features]
        all_features = torch.cat(batch_features, dim=0)
        all_query_features = self.query_net(all_features)
        all_key_features = self.key_net(all_features)

        batch_query_features, batch_key_features = [], []
        start = 0
        for nfeatures in batch_nfeatures:
            end = start + nfeatures
            query_features = all_query_features[start:end, :]
            key_features = all_key_features[start:end, :]
            start = end

            batch_query_features.append(query_features)
            batch_key_features.append(key_features)

        batch_similarity_mtxs = _get_batch_pairwise_dot_products(batch_query_features, batch_key_features)
        return batch_similarity_mtxs


def _get_batch_pairwise_dot_products(batch_features:list, batch_other_features: list):
    nbatch = len(batch_features)
    assert nbatch == len(batch_other_features)

    batch_dot_products = []
    for i in range(nbatch):
        features = batch_features[i]  # N x D
        other_features = batch_other_features[i]  # N x D
        assert features.shape == other_features.shape
        other_features = other_features.permute(1, 0)  # D x N
        dot_product = torch.matmul(features, other_features)  # N x N
        batch_dot_products.append(dot_product)
    return batch_dot_products
