import torch
import numpy as np

from tqdm.notebook import tqdm
from collections import defaultdict

def train(model, optimizer, loader, device, criterion=None):
    model.train()
    losses_tr = []
    for question, types, mask, start, end in tqdm(loader):
        optimizer.zero_grad()
        question = question.to(device)
        types = types.to(device)
        mask = mask.to(device)
        start = start.to(device)
        end = end.to(device)
        pred = model(question, types, mask, start, end)
        
        pred.loss.backward()
        optimizer.step()
        losses_tr.append(pred.loss.item())
    
    return model, optimizer, np.mean(losses_tr)

def check(pred_start, pred_end, real_start, real_end):
    return pred_start == real_start and pred_end == real_end

def val(model, loader, criterion, device, metric_names=None):
    model.eval()
    losses_val = []
    if metric_names is not None:
        metrics = defaultdict(list)
    with torch.no_grad():
        for question, types, mask, start, end in tqdm(loader):
            question = question.to(device)
            types = types.to(device)
            mask = mask.to(device)
            start = start.to(device)
            end = end.to(device)
            pred = model(question, types, mask, start, end)

            losses_val.append(pred.loss.item())
            pred_start = pred.start_logits
            pred_end = pred.end_logits
            if metric_names is not None:
                if 'accuracy' in metric_names:
                    preds_start = torch.argsort(pred_start, dim=1, descending=True)
                    for k in metric_names["accuracy"]["top"]:
                        metrics[f'start:accuracy ~ top#{k}'].append(
                            np.mean([start[i].item() in preds_start[i, :k] for i in range(start.shape[0])])
                        )
                    preds_end = torch.argsort(pred_end, dim=1, descending=True)
                    for k in metric_names["accuracy"]["top"]:
                        metrics[f'end:accuracy ~ top#{k}'].append(
                            np.mean([end[i].item() in preds_end[i, :k] for i in range(end.shape[0])])
                        )
                    metrics['accuracy'].append(
                            np.mean([check(
                                preds_start[i, 0].item(), 
                                preds_end[i, 0].item(), 
                                start[i].item(), 
                                end[i].item()
                            ) for i in range(end.shape[0])])
                    )

        if metric_names is not None:
            for name in metrics:
                metrics[name] = np.mean(metrics[name])
    
    return np.mean(losses_val), metrics if metric_names else None