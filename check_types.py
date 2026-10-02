import time
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pipeline.dataset import FloorPlanDataset
from torch_geometric.loader import DataLoader
import pipeline.config as cfg
import torch
from pipeline.model import UNet, GaussianDiffusion, GraphEncoder

with open(cfg.DATA_PKL, "rb") as f:
    import pickle
    all_plans = pickle.load(f)

train_ds = FloorPlanDataset(split="train", plans=all_plans)
train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=False)
batch = next(iter(train_loader))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch = batch.to(device)

denoise_model = UNet(
    in_channels=cfg.INPUT_CHANNELS, out_channels=cfg.INPUT_CHANNELS,
    base_channels=cfg.BASE_CHANNELS, ch_mults=(1, 2, 4, 8), num_res_blocks=2,
    time_emb_dim=256, cond_emb_dim=cfg.BASE_CHANNELS, dropout=cfg.DROPOUT
).to(device)

# We wrap denoise_model with torch.compile to see if the type issue occurs
denoise_model = torch.compile(denoise_model)

graph_encoder = GraphEncoder(
    in_dim=cfg.NUM_ROOM_TYPES + 1, hidden_dim=cfg.BASE_CHANNELS, num_layers=3, edge_dim=cfg.NUM_EDGE_TYPES
).to(device)

diffusion = GaussianDiffusion(
    model=denoise_model, image_size=cfg.IMAGE_SIZE, timesteps=cfg.TIMESTEPS,
    beta_start=cfg.BETA_START, beta_end=cfg.BETA_END
).to(device)

with torch.amp.autocast(device_type=device.type, enabled=True):
    global_emb, _ = graph_encoder(batch.x, batch.edge_index, batch.edge_attr if hasattr(batch, "edge_attr") else None)
    t = torch.randint(0, diffusion.timesteps, (batch.target_img.shape[0],), device=device).long()
    
    x_start = batch.target_img
    noise = torch.randn_like(x_start)
    x_noisy = diffusion.q_sample(x_start=x_start, t=t, noise=noise)
    
    print(f"global_emb dtype: {global_emb.dtype}")
    print(f"x_noisy dtype: {x_noisy.dtype}")
    print(f"t dtype: {t.dtype}")
    
    predicted_noise = denoise_model(x_noisy, t, graph_data=None, cond_emb=global_emb)

print("Done without errors.")
