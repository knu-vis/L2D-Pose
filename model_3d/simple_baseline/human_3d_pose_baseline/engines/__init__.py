import torch
from tqdm import tqdm

from ..evaluators import get_evaluator, get_evaluator_panoptic


def train_epoch(config, model, criterion, optimizer, lr_scheduler, human36m, device):
    """Train the model for an epoch.

    Args:
        config (yacs.config.CfgNode): Configuration.
        model (torch.nn.Module): Model to train.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optimizer): Optimizer for training
        lr_scheduler (torch.lr_sceduler): Learning scheduler for training.
        human36m (Human36MDatasetHandler): Human3.6M dataset.
        device (torch.device): CUDA device to use for training.

    Returns:
        (dict): training results.
    """
    model.train()

    sum_loss, num_samples = 0, 0

    for batch in tqdm(human36m.train_dataloader):
        data = batch["pose_2d"].to(device)
        target = batch["pose_3d"].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        batch_size = len(data)
        sum_loss += loss.item() * batch_size
        num_samples += batch_size

    average_loss = sum_loss / num_samples
    metrics = {"loss": average_loss}

    return metrics


def test_epoch(config, model, criterion, human36m, device):
    """Evaluate the model.

    Args:
        config (yacs.config.CfgNode): Configuration.
        model (torch.nn.Module): Model to test.
        criterion (torch.nn.Module): Loss function.
        human36m (Human36MDatasetHandler): Human3.6M dataset.
        device (torch.device): CUDA device to use for training.

    Returns:
        (dict): evaluation results.
    """
    evaluator = get_evaluator(config, human36m)  # Joint error evaluator.

    model.eval()

    sum_loss, num_samples = 0, 0

    with torch.no_grad():
        for batch in tqdm(human36m.test_dataloader):
            data = batch["pose_2d"].to(device)
            target = batch["pose_3d"].to(device)
            output = model(data)
            loss = criterion(output, target)

            batch_size = len(data)
            sum_loss += loss.item() * batch_size
            num_samples += batch_size

            # Joint error evaluation.
            action = batch["action"]  # Used for per action evaluation.
            evaluator.add_samples(
                pred_3d_poses=output.data.cpu().numpy(),
                truth_3d_poses=target.data.cpu().numpy(),
                actions=action,
            )

    metrics = evaluator.get_metrics()

    # Add average test loss to the metric dictionary.
    average_loss = sum_loss / num_samples
    assert "loss" not in metrics
    metrics["loss"] = average_loss

    return metrics




def train_epoch_panoptic(config, model, criterion, optimizer, lr_scheduler, panoptic, device):
    """Train the model for an epoch.

    Args:
        config (yacs.config.CfgNode): Configuration.
        model (torch.nn.Module): Model to train.
        criterion (torch.nn.Module): Loss function.
        optimizer (torch.optimizer): Optimizer for training
        lr_scheduler (torch.lr_sceduler): Learning scheduler for training.
        panoptic (Human36MDatasetHandler): Human3.6M dataset.
        device (torch.device): CUDA device to use for training.

    Returns:
        (dict): training results.
    """
    model.train()

    sum_loss, num_samples = 0, 0

    for batch in tqdm(panoptic.train_dataloader):
        data = batch["pose_2d"].to(device)
        target = batch["pose_3d"].to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        batch_size = len(data)
        sum_loss += loss.item() * batch_size
        num_samples += batch_size

    average_loss = sum_loss / num_samples
    metrics = {"loss": average_loss}

    return metrics


def test_epoch_panoptic(config, model, criterion, panoptic, device):
    """Evaluate the model.

    Args:
        config (yacs.config.CfgNode): Configuration.
        model (torch.nn.Module): Model to test.
        criterion (torch.nn.Module): Loss function.
        panoptic (Human36MDatasetHandler): Human3.6M dataset.
        device (torch.device): CUDA device to use for training.

    Returns:
        (dict): evaluation results.
    """
    evaluator = get_evaluator_panoptic(config, panoptic)  # Joint error evaluator.

    model.eval()

    sum_loss, num_samples = 0, 0

    with torch.no_grad():
        for batch in tqdm(panoptic.test_dataloader):
            data = batch["pose_2d"].to(device)
            target = batch["pose_3d"].to(device)
            output = model(data)
            loss = criterion(output, target)

            batch_size = len(data)
            sum_loss += loss.item() * batch_size
            num_samples += batch_size

            # Joint error evaluation.
            evaluator.add_samples(
                pred_3d_poses=output.data.cpu().numpy(),
                truth_3d_poses=target.data.cpu().numpy(),
            )

    metrics = evaluator.get_metrics()

    # Add average test loss to the metric dictionary.
    average_loss = sum_loss / num_samples
    assert "loss" not in metrics
    metrics["loss"] = average_loss

    return metrics
