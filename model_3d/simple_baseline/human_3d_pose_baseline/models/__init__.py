import torch

from ..utils.cuda import get_device
from .baseline_model import BaselineModel


def get_model(config, num_joints, model_weight_path=None):
    """Get model.

    Args:
        config (yacs.config.CfgNode): Configuration.

    Returns:
        (torch.nn.Module): Model.
    """
    print("Loading model...")

    model = BaselineModel(
        linear_size=config.MODEL.LINEAR_SIZE,
        num_stages=config.MODEL.NUM_STAGES,
        p_dropout=config.MODEL.DROPOUT_PROB,
        predict_14=config.MODEL.PREDICT_14,
        num_joints=num_joints
    )
    print(f"num_joints: {num_joints}")
    if model_weight_path:
        model.load_state_dict(torch.load(model_weight_path, map_location=torch.device("cpu")))
        print(f"Loaded weight from {model_weight_path}.")

    device = get_device(config.USE_CUDA)
    model = model.to(device)

    print("Done!")

    return model
