import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear, ReLU, Module
from torch_geometric.nn import MessagePassing, BatchNorm
from torch_geometric.nn.aggr import SumAggregation, MeanAggregation, SortAggregation, Set2Set


class AggrFactory:
    def __init__(self):
        self.aggr_types = {
            'sum': SumAggregation,
            'mean': MeanAggregation,
            'sort': SortAggregation,
            'set2set': Set2Set,
        }

    def factory(self, type, **kwargs):
        """
        Factory method for creating aggregation functions.

        Args:
            type (str): The type of aggregation function to create.
            **kwargs: Additional keyword arguments to pass to the aggregation function.

        Returns:
            An instance of the specified aggregation function.

        Raises:
            ValueError: If an invalid aggregation type is specified or if the specified type
                does not accept the provided keyword arguments.
        """
        try:
            fun = self.aggr_types[type]
            return fun(**kwargs)
        except KeyError:
            raise ValueError(f"Invalid aggregation type: {type}")
        except TypeError:
            raise ValueError(f"Invalid arguments for aggregation type: {type}")

    def dl_in_channels_factor(self, type, **kwargs):
        """
        Computes an adjustment factor for each aggregation type.

        Args:
            type (str): The type of aggregation to be performed. Supported types are "sum", "mean", "sort", and "set2set".
            **kwargs: Additional keyword arguments that may be required for certain aggregation types.

        Returns:
            int: an adjustment factor for the given aggregation type.

        Raises:
            ValueError: If an invalid aggregation type is provided.
        """
        if type == "sum":
            return 1
        elif type == "mean":
            return 1
        elif type == "sort":
            return kwargs["k"]
        elif type == "set2set":
            return 2
        else:
            raise ValueError(f"Invalid aggregation type: {type}")


# implementation of "Dynamic Graph CNN for Learning on Point Clouds" paper in pytorch geometric
class EdgeConv(MessagePassing):
    # N nodes
    # E edges
    # in_channels = features
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='max')  # "Max" aggregation.

        # building multi-layer perceptron
        self.mlp = Seq(Linear(2 * in_channels, out_channels),
                       ReLU(),
                       Linear(out_channels, out_channels))

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, x=x)

    # build all messages to node i
    def message(self, x_i, x_j):
        # i is the node that aggregates information
        # j is its neighbour

        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels] 

        # concatenates tensors
        tmp = torch.cat([x_i, x_j - x_i], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.mlp(tmp)


class MPGNN(Module):

    def __init__(self, dataset, num_layers, dropout, num_neurons, k):
        super().__init__()

        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")
        self.num_features = dataset.num_features
        self.num_output_features = dataset.num_classes
        self.num_neurons = num_neurons
        self.graph_conv_layer_sizes = num_layers
        self.aggr_type = "sort"
        self.aggr_args = {'k': k}
        self.dropout_rate = dropout

        self.conv1 = EdgeConv(self.num_features, self.num_neurons).to(self.device)
        self.batchnorm1 = BatchNorm(self.num_neurons).to(self.device)

        self.convs = torch.nn.ModuleList()
        self.batchnorms = torch.nn.ModuleList()

        for i in range(num_layers - 1):
            self.convs.append(EdgeConv(num_neurons, num_neurons).to(self.device))
            self.batchnorms.append(BatchNorm(num_neurons).to(self.device))

        aggr_factory = AggrFactory()
        self.aggr_layer = aggr_factory.factory(self.aggr_type, **self.aggr_args)

        # In forward, there is some tensor magic between these two "layers"
        self.aggr_factor = aggr_factory.dl_in_channels_factor(self.aggr_type, **self.aggr_args)

        first_linear_size = self.aggr_factor * self.num_neurons
        self.linear = torch.nn.Linear(first_linear_size, int(num_neurons/2)).to(self.device)
        self.linear_output = Linear(int(num_neurons/2), self.num_output_features).to(self.device)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = x.to(self.device)
        edge_index = edge_index.to(self.device)
        batch = batch.to(self.device)

        # Graph Convolutions
        x = self.conv1(x, edge_index)
        x = self.batchnorm1(x)
        x = F.leaky_relu(x)

        for conv, batchn in zip(self.convs, self.batchnorms):
            x = conv(x, edge_index)
            x = batchn(x)
            x = F.leaky_relu(x)

        # Aggregation
        x = self.aggr_layer(x, batch)

        # Reshape the tensor to be able to pass it to the dense layers (flatten ?)
        x = x.view(len(x), -1)

        # Dropout
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.linear(x)
        x = F.leaky_relu(x)
        x = self.linear_output(x)
        return x
