import torch
from tqdm import tqdm

def train(edge_index, loader, model, optimizer, num_nodes):
    total_loss = total_examples = 0

    for index in loader:
        pos_edge_label_index = edge_index[:, index]
        neg_edge_label_index = torch.stack([
            pos_edge_label_index[0],
            torch.randint(num_nodes[0], num_nodes[0] + num_nodes[1],
                          (index.numel(), ))
        ], dim=0)
        edge_label_index = torch.cat([
            pos_edge_label_index,
            neg_edge_label_index,
        ], dim=1)

        optimizer.zero_grad()
        pos_rank, neg_rank = model(edge_index, edge_label_index).chunk(2)
        loss = model.recommendation_loss(
            pos_rank,
            neg_rank,
            node_id=edge_label_index.unique(),
        )

        loss.backward()
        optimizer.step()

        total_loss += float(loss) * pos_rank.numel()
        total_examples += pos_rank.numel()

    return total_loss / total_examples