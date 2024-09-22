# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 17:31:07 2024

@author: seyed.mousavi
"""
%reset -f
import os
import torch
import numpy as np
from torch_geometric.data import DataLoader, Data, InMemoryDataset
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from bvh import Bvh


# Global definitions
joint_names = [
    "Hips", "LHipJoint", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase",
    "RHipJoint", "RightUpLeg", "RightLeg", "RightFoot", "RightToeBase",
    "LowerBack", "Spine", "Spine1", "Neck", "Neck1", "Head", "LeftShoulder",
    "LeftArm", "LeftForeArm", "LeftHand", "LeftFingerBase", "LeftHandIndex1",
    "LThumb", "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    "RightFingerBase", "RightHandIndex1", "RThumb"
]

skeletal_connections = [
    ("Hips", "LHipJoint"), ("LHipJoint", "LeftUpLeg"), ("LeftUpLeg", "LeftLeg"), 
    ("LeftLeg", "LeftFoot"), ("LeftFoot", "LeftToeBase"),
    ("Hips", "RHipJoint"), ("RHipJoint", "RightUpLeg"), ("RightUpLeg", "RightLeg"), 
    ("RightLeg", "RightFoot"), ("RightFoot", "RightToeBase"),
    ("Hips", "LowerBack"), ("LowerBack", "Spine"), ("Spine", "Spine1"), 
    ("Spine1", "Neck"), ("Neck", "Neck1"), ("Neck1", "Head"),
    ("Spine1", "LeftShoulder"), ("LeftShoulder", "LeftArm"), 
    ("LeftArm", "LeftForeArm"), ("LeftForeArm", "LeftHand"), 
    ("LeftHand", "LeftFingerBase"), ("LeftFingerBase", "LeftHandIndex1"), 
    ("LeftHand", "LThumb"),
    ("Spine1", "RightShoulder"), ("RightShoulder", "RightArm"), 
    ("RightArm", "RightForeArm"), ("RightForeArm", "RightHand"), 
    ("RightHand", "RightFingerBase"), ("RightFingerBase", "RightHandIndex1"), 
    ("RightHand", "RThumb")
]

# Load BVH data
def load_bvh(file_path):
    with open(file_path) as f:
        mocap = Bvh(f.read())
    joint_motion_data = {joint: [] for joint in joint_names}
    num_frames = mocap.nframes
    for frame_number in range(num_frames):
        for joint_name in joint_names:
            channels_data = []
            for channel in ['Xposition', 'Yposition', 'Zposition', 'Xrotation', 'Yrotation', 'Zrotation']:
                if channel in mocap.joint_channels(joint_name):
                    channel_data = mocap.frame_joint_channels(frame_number, joint_name, [channel])
                    channels_data.append(float(channel_data[0]))
                else:
                    channels_data.append(0.0)
            joint_motion_data[joint_name].append(channels_data)
    return joint_motion_data, num_frames

# Interpolate motion data to match maximum number of frames
def interpolate_motion_data(joint_motion_data, num_frames, max_frames):
    interpolated_data = {}
    for joint_name, channel_data in joint_motion_data.items():
        # Ensuring channel_data is a NumPy array for easy manipulation
        channel_data = np.array(channel_data)
        
        # Initialize an array to hold interpolated data for this joint
        interpolated_channel_data = np.zeros((max_frames, channel_data.shape[1]))
        
        # Interpolate each dimension (column) individually
        for i in range(channel_data.shape[1]):
            # Apply 1D linear interpolation for each dimension
            interpolated_channel_data[:, i] = np.interp(
                np.linspace(0, num_frames - 1, max_frames),  # Target frames
                np.arange(num_frames),  # Original frames
                channel_data[:, i]  # Original data for this dimension
            )
        
        # Store the interpolated data
        interpolated_data[joint_name] = interpolated_channel_data
    
    return interpolated_data


# Skeleton to Graph-------------------------------------------------------------------------
def create_graph_for_all_frames(joint_motion_data, joint_names, max_frames):

    edge_index = []
    edge_weight = []

    # Dynamically assign weights
    for s, t in skeletal_connections:
        s_index = joint_names.index(s) # Find the index of the starting joint
        t_index = joint_names.index(t) # Find the index of the ending joint
        # using the index difference as a proxy for distance; larger differences imply further apart in the hierarchy
        weight = abs(s_index - t_index) # Calculate weight based on index difference
        edge_index.append([s_index, t_index])
        edge_weight.append(weight)

    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous() #Contains the indices of the start and end nodes for each edge, essential for defining the graph's connectivity.
    edge_weight_tensor = torch.tensor(edge_weight, dtype=torch.float) #Contains the weights of each edge, potentially impacting how signals are passed in the network during training.
    
    node_features = []
    for joint in joint_names:
        joint_data = np.array(joint_motion_data[joint]).flatten()
        node_features.append(joint_data)
    node_features_tensor = torch.tensor(node_features, dtype=torch.float)
    
    graph = Data(x=node_features_tensor, edge_index=edge_index_tensor, edge_attr=edge_weight_tensor.view(-1, 1))
    
    print("Edge Weights:", edge_weight_tensor.tolist())
    return graph


# -------------------------------------------------------------------------
# Dataset class for BVH files
class BVHGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(BVHGraphDataset, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        self.graph_objects = []  # initialize the storage for graph objects

    @property
    def processed_file_names(self):
        return ['data.pt']

    def process(self):
        data_list = []
        self.graph_objects = []   
        max_frames = 0
        classes = ['class1', 'class2', 'class3', 'class4', 'class5']
        for cls_index, cls_name in enumerate(classes):
            cls_path = os.path.join(self.root, cls_name)
            for file_name in os.listdir(cls_path):
                if not file_name.endswith('.bvh'): continue
                file_path = os.path.join(cls_path, file_name)
                joint_motion_data, num_frames = load_bvh(file_path)
                max_frames = max(max_frames, num_frames)
        for cls_index, cls_name in enumerate(classes):
            cls_path = os.path.join(self.root, cls_name)
            for file_name in os.listdir(cls_path):
                if not file_name.endswith('.bvh'): continue
                file_path = os.path.join(cls_path, file_name)
                joint_motion_data, num_frames = load_bvh(file_path)
                interpolated_data = interpolate_motion_data(joint_motion_data, num_frames, max_frames)
                graph = create_graph_for_all_frames(interpolated_data, joint_names, max_frames)
                graph.y = torch.tensor([cls_index])
                self.graph_objects.append(graph)  # Append each graph object here
                data_list.append(graph)
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


# GCN classification-------------------------------------------------------------------------
# Defining the GCN model

class GCN(torch.nn.Module):
    def __init__(self, num_node_features, num_classes):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_node_features, 16)
        self.conv2 = GCNConv(16, num_classes)

    def forward(self, data, return_features=False):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        if return_features:
            return x
        x = global_mean_pool(x, batch)
        return F.log_softmax(x, dim=1)



# collectting nodes and edges after training
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
def collect_features(model, loader):
    model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for data in loader:
            out_features = model(data, return_features=True)  # Collect features
            features.append(out_features.cpu().numpy())
            labels.extend(data.y.cpu().numpy())
    return np.concatenate(features, axis=0), labels



#-------------------------------------------------------------------------
# Graph Data
import torch
# Load the processed data
processed_data_path = 'Actions Five Main/processed/data.pt'   
data = torch.load(processed_data_path)
# 'data' is a list or iterable of dictionaries representing graph data
nodes_features_samples = []
edges_list_samples = []

# inspecting the first 3 graphs as an example
num_samples_to_inspect = 5
for i in range(min(num_samples_to_inspect, len(data))):
    graph_dict = data[i]   
    if 'x' in graph_dict and 'edge_index' in graph_dict:
        nodes_features_samples.append(graph_dict['x'])  # Access node features
        edges_list_samples.append(graph_dict['edge_index'])  # Access edge list
    else:
        print(f"Graph {i} does not have 'x' or 'edge_index' keys")
# Output the variables to inspect them
print("Node features of the first 3 graphs:", nodes_features_samples)
print("Edge lists of the first 3 graphs:", edges_list_samples)



# Extracting Nodes and Edges
tensor_at_index_zero = nodes_features_samples[0]
NodeF = tensor_at_index_zero.numpy()
lbl = nodes_features_samples[1]
lbl1 = lbl.numpy()
tensor_at_index_zero_edge = edges_list_samples[0]
EdgeF = tensor_at_index_zero_edge.numpy()

# Labels
C_Walk = [0] * 135
C_Jump = [1] * 18
C_Kick = [2] * 25
C_Punch = [3] * 44
C_Run = [4] * 30
# Concatenate
Labels = C_Walk + C_Jump + C_Kick + C_Punch + C_Run
Labels_int32= np.array(Labels, dtype=np.int32)

# Normalizing labels into nodes size
import numpy as np
unique, counts = np.unique(Labels_int32, return_counts=True)
proportions = counts / counts.sum()
# Calculate the number of rows in NodeF
num_rows = NodeF.shape[0]
# Calculate target counts for each category based on the number of rows in NodeF
target_counts = np.round(proportions * num_rows).astype(int)
target_counts[-1] = num_rows - target_counts[:-1].sum()
# Create the new labels array
new_labels = np.hstack([np.full(count, category, dtype=np.int32) for category, count in zip(unique, target_counts)])
print("Distribution of new labels:", np.unique(new_labels, return_counts=True))


# # Plot t-SNE with
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# tsne = TSNE(n_components=2, perplexity=400, learning_rate=2, n_iter=1500, random_state=42)
# tsne_results = tsne.fit_transform(NodeF)
# plt.figure(figsize=(10, 8))
# colors = ['orange', 'green', 'blue', 'purple', 'red']
# category_names = ['Walk', 'Jump', 'Kick', 'Punch', 'Run'] 
# categories = np.unique(new_labels) 
# # Plot each category
# for i, category in enumerate(categories):
#     idx = new_labels == category
#     plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1], c=colors[i], label=f'{category_names[i]}', alpha=0.5)
# plt.title('Adjusted t-SNE Plot of Nodes', fontsize=18)  
# plt.xlabel('t-SNE Axis 1', fontsize=15)  
# plt.ylabel('t-SNE Axis 2', fontsize=15)  
# plt.legend(title='Activity', title_fontsize='15', fontsize='14')  
# plt.grid(True)
# plt.show()


#-------------------------------------------------------------------------
# Train and test
import random
import torch
from sklearn.metrics import classification_report, confusion_matrix
def collate_fn(data_list):
    return Batch.from_data_list(data_list)
import matplotlib.pyplot as plt
import torch


def split_dataset(dataset, train_ratio=0.7):
    n = len(dataset)
    indices = list(range(n))
    random.shuffle(indices)

    split = int(np.floor(train_ratio * n))
    train_indices, test_indices = indices[:split], indices[split:]

    train_dataset = [dataset[i] for i in train_indices]
    test_dataset = [dataset[i] for i in test_indices]

    return train_dataset, test_dataset

def train(model, loader, optimizer):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for data in loader:
        optimizer.zero_grad()
        out = model(data)
        loss = F.nll_loss(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        _, predicted = torch.max(out, 1)
        total += data.y.size(0)
        correct += (predicted == data.y).sum().item()
    avg_loss = total_loss / len(loader)
    accuracy = correct / total
    return avg_loss, accuracy

# Define class index to name mapping
class_names = {0: 'Walk', 1: 'Jump', 2: 'Kick', 3: 'Punch', 4: 'Run'}

def test(model, loader, print_metrics=False):
    model.eval()
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for data in loader:
            out = model(data)
            _, predicted = torch.max(out, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(data.y.cpu().numpy())
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    if print_metrics:
        # Replace numeric labels with class names using the mapping
        all_labels = [class_names[label] for label in all_labels]
        all_preds = [class_names[pred] for pred in all_preds]
        
        print('Classification Report:')
        print(classification_report(all_labels, all_preds, zero_division=0, labels=list(class_names.values())))
        print('Confusion Matrix:')
        print(confusion_matrix(all_labels, all_preds, labels=list(class_names.values())))
    return accuracy



# Extracting nodes and edges after training
def extract_features_and_edges(model, loader):
    model.eval()
    all_node_features = []
    all_edge_indices = []
    with torch.no_grad():
        for data in loader:
            node_features = model(data, return_features=True)  # Extract features before pooling
            all_node_features.append(node_features.cpu().numpy())
            all_edge_indices.append(data.edge_index.cpu().numpy())
    return all_node_features, all_edge_indices




import torch
from torch_geometric.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

def main():
    dataset = BVHGraphDataset(root='Actions Five Main')
    train_dataset, test_dataset = split_dataset(dataset, train_ratio=0.7)
    
    train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)
    
    num_classes = 5
    num_node_features = dataset.num_node_features
    model = GCN(num_node_features, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, weight_decay=5e-4)

    train_losses = []
    train_accs = []
    test_accs = []

    for epoch in range(100):
        train_loss, train_acc = train(model, train_loader, optimizer)
        test_acc = test(model, test_loader)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)

        print(f'Epoch: {epoch}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')

    # Extract node features and edge indices 
    NodeFAfter, EdgeFAfter = extract_features_and_edges(model, test_loader)

    # Final metrics
    final_test_accuracy = test(model, test_loader, print_metrics=True)
    print(f'Final Test Accuracy: {final_test_accuracy:.4f}')

    # Plotting the training and testing metrics
    plt.rcParams['font.size'] = 14  
    plt.rcParams['font.weight'] = 'bold'  
    plt.rcParams['axes.labelweight'] = 'bold'  
    plt.rcParams['axes.titleweight'] = 'bold' 
    plt.rcParams['lines.linewidth'] = 2  
    fig, axs = plt.subplots(2, 1, figsize=(10, 10))  # Create 2 plots vertically
    # Plot training loss
    axs[0].plot(train_losses, 'r-', label='Train Loss')
    axs[0].set_title('Training Loss over Epochs')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].legend()
    # Plot training and test accuracy
    axs[1].plot(train_accs, 'b--', label='Train Accuracy')
    axs[1].plot(test_accs, 'g-.', label='Test Accuracy')
    axs[1].set_title('Training and Test Accuracy over Epochs')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()
    plt.tight_layout()
    plt.show()

    return NodeFAfter, EdgeFAfter  # Return the extracted features and edges

if __name__ == "__main__":
    node_features_after, edge_indices_after = main()




# # TSNE after training

# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.manifold import TSNE
# # Data generation
# node_features_after = [np.random.rand(155, 5) for _ in range(16)]
# # Extract dimensions dynamically
# num_batches = len(node_features_after)
# nodes_per_batch = node_features_after[0].shape[0]
# features_per_node = node_features_after[0].shape[1]
# total_samples = num_batches * nodes_per_batch
# total_features = num_batches * features_per_node
# # empty array to hold the final dataset
# full_dataset = np.zeros((total_samples, total_features))
# # Fill the dataset
# for batch_index, batch in enumerate(node_features_after):
#     for node_index in range(nodes_per_batch):
#         sample_index = batch_index * nodes_per_batch + node_index
#         start_pos = batch_index * features_per_node
#         end_pos = start_pos + features_per_node
#         full_dataset[sample_index, start_pos:end_pos] = batch[node_index, :]
# # shape of full_dataset
# print("Shape of full_dataset:", full_dataset.shape)
# # Labels setup corrected
# labels_per_category = total_samples // 5  # Evenly distribute labels
# remainder = total_samples % 5  # Calculate any leftover samples
# # Creating labels with possible adjustments for remainder
# labels = []
# categories = [0, 1, 2, 3, 4]  # Representing Walk, Jump, Kick, Punch, Run
# for i, category in enumerate(categories):
#     labels.extend([category] * (labels_per_category + (1 if i < remainder else 0)))
# # Convert labels to numpy array
# Labels = np.array(labels)
# # Ensure total labels match node features
# assert len(Labels) == full_dataset.shape[0], "Label count does not match node count!"
# # Run t-SNE
# tsne = TSNE(n_components=2, perplexity=4, learning_rate=200, n_iter=500, random_state=42)
# tsne_results = tsne.fit_transform(full_dataset)
# # Plotting
# plt.figure(figsize=(10, 8))
# colors = ['orange', 'green', 'blue', 'purple', 'red']
# category_names = ['Walk', 'Jump', 'Kick', 'Punch', 'Run']
# categories = np.unique(Labels)
# for i, category in enumerate(categories):
#     idx = Labels == category
#     plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1], c=colors[i], label=category_names[i], alpha=0.5)
# plt.title('t-SNE Visualization of Node Features')
# plt.xlabel('t-SNE Dimension 1')
# plt.ylabel('t-SNE Dimension 2')
# plt.legend(title='Activity', title_fontsize='13', fontsize='12')
# plt.grid(True)
# plt.show()







# #-------------------------------------------------------------------------
# # Plot one body graph but random shape
# import networkx as nx
# import matplotlib.pyplot as plt
# from torch_geometric.utils import to_networkx
# # List of joint names
# def plot_graph_from_dataset(dataset, graph_index=0):
#     # Access the graph Data object
#     graph_data = dataset[graph_index]
#     # Convert to NetworkX graph
#     G = to_networkx(graph_data, to_undirected=True)    
#     # Optional: Add node labels based on joint names
#     if len(joint_names) == len(G.nodes()):
#         node_labels = {i: f'{i}: {joint_names[i]}' for i in range(len(G.nodes()))}
#     else:
#         # Fallback to using node indices as labels if there's a mismatch
#         node_labels = {i: str(i) for i in range(len(G.nodes()))}    
#     # Draw the graph
#     plt.figure(figsize=(12, 8))
#     nx.draw(G, with_labels=True, labels=node_labels, node_size=700, node_color="lightblue", font_size=10)
#     plt.title(f"Graph Representation of Body Motion Sample {graph_index}")
#     plt.show()
# # Example usage
# dataset = BVHGraphDataset(root='Emotions Four Main')   
# plot_graph_from_dataset(dataset, graph_index=1)  # Visualize the specified graph with joint names



# #-------------------------------------------------------------------------
# # Plot one body graph with starting shape
# import networkx as nx
# import matplotlib.pyplot as plt
# from torch_geometric.utils import to_networkx
# # joint_names is a list of joint names in the same order as nodes in graph
# def plot_graph_from_dataset(dataset, graph_index=0, seed=42):
#     # Access the graph Data object
#     graph_data = dataset[graph_index]
#     # Convert to NetworkX graph
#     G = to_networkx(graph_data, to_undirected=True)    
#     # Compute the positions of the nodes using a fixed seed for consistency
#     pos = nx.spring_layout(G, seed=seed)  # Use spring layout with a fixed seed    
#     # Add node labels based on joint names
#     # Check if the length of joint_names matches the number of nodes
#     if len(joint_names) == len(G.nodes()):
#         node_labels = {i: f'{i}: {joint_names[i]}' for i in range(len(G.nodes()))}
#     else:
#         # Fallback to using node indices as labels if there's a mismatch
#         node_labels = {i: str(i) for i in range(len(G.nodes()))}    
#     # Draw the graph using the precomputed positions
#     plt.figure(figsize=(12, 8))
#     nx.draw(G, pos, with_labels=True, labels=node_labels, node_size=700, node_color="lightblue", font_size=10)
#     plt.title(f"Graph Representation of Body Motion Sample {graph_index}")
#     plt.show()
# # Example usage
# dataset = BVHGraphDataset(root='Emotions Four Main')   
# plot_graph_from_dataset(dataset, graph_index=6)  # Visualize the specified graph with consistent layout




# #-------------------------------------------------------------------------
# #  Plot one body graph for first 8 samples random shape
# import matplotlib.pyplot as plt
# import networkx as nx
# from torch_geometric.utils import to_networkx
# def plot_graph_samples_subplots(dataset, indices):
#     # Set up the subplot grid
#     fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(24, 12))
#     axes = axes.flatten()  # Flatten to 1D array for easy iteration
#     for ax, graph_index in zip(axes, indices):
#         graph_data = dataset[graph_index]
#         G = to_networkx(graph_data, to_undirected=True)  
#         # customizing node labels
#         node_labels = {i: '' for i in range(len(G.nodes()))}  # Empty labels
#         # Draw the graph in its subplot
#         nx.draw(G, ax=ax, with_labels=True, labels=node_labels, node_size=50, node_color="lightblue", font_size=8)
#         ax.set_title(f"Sample {graph_index}")
#     plt.tight_layout()
#     plt.show()
# # Example usage:
# dataset = BVHGraphDataset(root='Emotions Four Main')   
# plot_graph_samples_subplots(dataset, indices=range(8))  # Visualize the first 8 samples



# #-------------------------------------------------------------------------
# # Plot one body graph with node and edge features
# import networkx as nx
# import matplotlib.pyplot as plt
# from torch_geometric.utils import to_networkx

# def plot_graph_from_dataset_with_features(dataset, graph_index=0, seed=42, display_node_features=False, display_edge_features=False):
#     # Access the graph Data object
#     graph_data = dataset[graph_index]
#     # Convert to NetworkX graph
#     G = to_networkx(graph_data, to_undirected=True)
#     # Compute the positions of the nodes using a fixed seed for consistency
#     pos = nx.spring_layout(G, seed=seed)  # Use spring layout with a fixed seed

#     plt.figure(figsize=(12, 8))  
    
#     # Draw the graph (nodes and edges)
#     nx.draw(G, pos, node_size=700, node_color="lightblue", font_size=10, with_labels=False)  # Hide default labels
    
#     # Node labels with joint name (and node feature if applicable)
#     node_labels = {}
#     for i in G.nodes():
#         # Prefer joint name if available; fallback to node number
#         label = joint_names[i] if 'joint_names' in locals() and len(joint_names) > i else f'Node {i}'
#         if display_node_features and 'x' in graph_data:
#             feature_value = graph_data.x[i][0].item()
#             label += f'\nFeat: {feature_value:.2e}'
#         node_labels[i] = label
#     nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

#     # Annotate edges with features if requested, in smaller font
#     if display_edge_features and 'edge_attr' in graph_data:
#         edge_labels = {}
#         for edge, feature in zip(graph_data.edge_index.t().numpy(), graph_data.edge_attr):
#             # display the first feature of each edge
#             edge_key = (edge[0], edge[1])
#             edge_labels[edge_key] = f'{feature[0].item():.2e}'
#         nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, font_color='red')
    
#     plt.title(f"Graph Representation of Body Motion Sample {graph_index}")
#     plt.show()

# # Ensuring 'joint_names' is defined in environment before calling this function
# # Edataset and joint_names are properly initialized:
# plot_graph_from_dataset_with_features(dataset, graph_index=6, display_node_features=True, display_edge_features=True)


