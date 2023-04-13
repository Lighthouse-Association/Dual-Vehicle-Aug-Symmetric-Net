import torch
from torchmetrics import RetrievalMRR


def evaluate_retrieval_results(gt_tracks, results):
    """
    Compute evaluation metrics for the baseline model.
    """
    recall_5 = 0
    recall_10 = 0
    mrr = 0
    for query in gt_tracks:
        result = results[query]
        target = gt_tracks[query]
        try:
            rank = result.index(target)
        except ValueError:
            rank = 100
        if rank < 10:
            recall_10 += 1
        if rank < 5:
            recall_5 += 1
        mrr += 1.0 / (rank + 1)
    recall_5 /= len(gt_tracks)
    recall_10 /= len(gt_tracks)
    mrr /= len(gt_tracks)
    return mrr, recall_5, recall_10

def get_mrr(sim_mat):
    mrr = RetrievalMRR()
    return mrr(
        sim_mat.flatten(),
        torch.eye(len(sim_mat), device=sim_mat.device).long().bool().flatten(),
        torch.arange(len(sim_mat), device=sim_mat.device)[:, None].expand(len(sim_mat), len(sim_mat)).flatten(),
    )
    pass

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        # pred(correct.shape)
        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
