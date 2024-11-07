import torch
from torch_geometric.utils import degree

@torch.no_grad()
def test(edge_index, model, num_nodes, batch_size, k):
    emb = model.get_embedding(edge_index)
    user_emb, book_emb = emb[:num_nodes[0]], emb[num_nodes[0]:]

    precision = recall = total_examples = 0
    for start in range(0, num_nodes[0], batch_size):
        end = start + batch_size
        logits = user_emb[start:end] @ book_emb.t()

        mask = ((edge_index[0] >= start) &
                (edge_index[0] < end))
        logits[edge_index[0, mask] - start,
               edge_index[1, mask] - num_nodes[0]] = float('-inf')

        ground_truth = torch.zeros_like(logits, dtype=torch.bool)
        mask = ((edge_index[0] >= start) &
                (edge_index[0] < end))
        ground_truth[edge_index[0, mask] - start,
                     edge_index[1, mask] - num_nodes[0]] = True
        node_count = degree(edge_index[0, mask] - start,
                            num_nodes=logits.size(0))

        topk_index = logits.topk(k, dim=-1).indices
        isin_mat = ground_truth.gather(1, topk_index)

        precision += float((isin_mat.sum(dim=-1) / k).sum())
        recall += float((isin_mat.sum(dim=-1) / node_count.clamp(1e-6)).sum())
        total_examples += int((node_count > 0).sum())

    return precision / total_examples, recall / total_examples