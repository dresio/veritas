from ik_model import IKNet
import torch

model = IKNet()
model.load_state_dict(torch.load("checkpoints/ik_model_rl_v2_1000.pt"))
model.eval()