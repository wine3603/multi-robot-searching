import torch
import numpy as np


def compute_iou(outputs, targets, threshold=0.5):
    """Compute mean IoU for occupancy channel"""
    if outputs.dim() == 3:  # [2, H, W]
        occ_pred = (outputs[0, :, :] > threshold).cpu().numpy()
    elif outputs.dim() == 4:  # [B, 2, H, W]
        occ_pred = (outputs[:, 0, :, :] > threshold).cpu().numpy()
    else:
        return 0.0

    if targets.dim() == 3:  # [2, H, W]
        occ_gt = (targets[0, :, :] > 0.5).cpu().numpy()
    elif targets.dim() == 4:  # [B, 2, H, W]
        occ_gt = (targets[:, 0, :, :] > 0.5).cpu().numpy()
    else:
        return 0.0

    intersection = np.logical_and(occ_pred, occ_gt).sum()
    union = np.logical_or(occ_pred, occ_gt).sum()
    if union == 0:
        return 0.0
    return intersection / union


# Training Function
def train_one_epoch(dataloader, model, loss_fn, optimizer, device, writer=None) -> float:
    num_batches = len(dataloader.dataset)
    train_loss = 0.0
    model.train()
    # for every batch_sized chunk of data ...
    for batch_idx, (samples, targets) in enumerate(dataloader):
        # copy data to the computing device (normally the GPU)
        samples = samples.to(device)
        targets = targets.to(device)
        # Compute prediction a prediction
        outputs = model(samples)
        # Compute the error (loss) of that prediction [loss_fn(prediction, target)]
        loss, losses_monitoring = loss_fn(outputs, targets)
        train_loss += loss.item()
        # Backpropagation strategy/ optimization "zero_grad()"
        optimizer.zero_grad()
        # Apply the prediction loss (backpropagation)
        loss.backward()
        # write the weights to the NN
        optimizer.step()
        if batch_idx % 100 == 0:
            loss, current = loss.item(), batch_idx * len(samples)
            losses_output = " ".join([f"{loss_mon['name']}: {loss_mon['value']:>7f}," for loss_mon in losses_monitoring])
            print(f"loss: {loss:>7f}  [{current:>5d}/{num_batches:>5d}], \t {losses_output}")
    train_loss /= num_batches
    if writer:
        # Log the average loss for the epoch
        writer.log({"Train loss": train_loss})
    return train_loss


# Validation Function
def evaluate(dataloader, model, loss_fn, device, epoch, writer=None):
    num_batches = len(dataloader)
    model.eval()
    eval_loss = 0
    iou_sum = 0.0
    count = 0
    with torch.no_grad():
        for (samples, targets) in dataloader:
            samples = samples.to(device)
            targets = targets.to(device)
            outputs = model(samples)
            # Handle different dimensions
            if outputs.dim() == 3 and targets.dim() == 3:  # batch size 1 after squeeze
                outputs = outputs.unsqueeze(0)
                targets = targets.unsqueeze(0)
            loss, _ = loss_fn(outputs, targets)
            eval_loss += loss
            # Compute IoU every epoch
            iou = compute_iou(outputs, targets, threshold=0.1)
            iou_sum += iou
            count += 1
    eval_loss /= num_batches
    avg_iou = iou_sum / count
    print(f"Test Error: \n Avg loss: {eval_loss:>8f}")
    print(f" Mean IoU (threshold=0.1): {avg_iou:>8f}\n")
    if writer:
        # Log the loss by adding scalars
        writer.log({"Eval loss": eval_loss})
        writer.log({"Eval IoU": avg_iou})
    return eval_loss, avg_iou


class EarlyStopping:
    def __init__(self, tolerance=5, min_delta=0.0):
        self.tolerance = tolerance
        self.min_delta = min_delta
        self.validation_loss_minus_one = 10000
        self.counter = 0
        self.early_stop = False

    def check_stop(self, validation_loss):
        if (self.validation_loss_minus_one - validation_loss) > self.min_delta:
            self.counter = 0
        else:
            self.counter += 1
        if self.counter >= self.tolerance:
            self.early_stop = True
        self.validation_loss_minus_one = validation_loss
