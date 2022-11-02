import torch
from tqdm import tqdm
import time


def train(optimizer, model, dataloader, loss_fn) -> dict:
    """
    Train models in an epoch.
    """
    model.train()
    total_loss = []
    for batch in dataloader:
        optimizer.zero_grad()
        pred = model(*batch[:-1], id=0)
        loss = loss_fn(pred, batch[-1])
        loss.backward()
        total_loss.append(loss.detach().item())
        optimizer.step()
    return {
        "loss": sum(total_loss) / len(total_loss),
        "num_batches": len(total_loss),
    }


@torch.no_grad()
def test(model, dataloader, metrics, loss_fn, return_dict=False):
    """
    Test models either on validation dataset or test dataset.
    """
    model.eval()
    preds = []
    ys = []
    for batch in dataloader:
        pred = model(*batch[:-1])
        preds.append(pred)
        ys.append(batch[-1])
    pred = torch.cat(preds, dim=0)
    y = torch.cat(ys, dim=0)
    if not return_dict:
        return metrics(pred.cpu().numpy(), y.cpu().numpy()), loss_fn(pred, y)
    else:
        return {
            "score": metrics(pred.cpu().numpy(), y.cpu().numpy()),
            "loss": loss_fn(pred, y),
            "num_batches": len(ys),
        }

