import time
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from pipeline.dataset import FloorPlanDataset
from torch_geometric.loader import DataLoader
import pipeline.config as cfg
import torch
from pipeline.model import UNet, GaussianDiffusion, GraphEncoder

print("Loading dataset...")
import pickle
with open(cfg.DATA_PKL, "rb") as f:
    all_plans = pickle.load(f)

train_ds = FloorPlanDataset(split="train", plans=all_plans)
train_loader = DataLoader(train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=min(4, os.cpu_count()))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cond_emb_dim = cfg.BASE_CHANNELS

denoise_model = UNet(
    in_channels=cfg.INPUT_CHANNELS,
    out_channels=cfg.INPUT_CHANNELS,
    base_channels=cfg.BASE_CHANNELS,
    ch_mults=(1, 2, 4, 8),
    num_res_blocks=2,
    time_emb_dim=256,
    cond_emb_dim=cond_emb_dim,
    dropout=cfg.DROPOUT
).to(device)

graph_encoder = GraphEncoder(
    in_dim=cfg.NUM_ROOM_TYPES + 1,
    hidden_dim=cond_emb_dim,
    num_layers=3,
    edge_dim=cfg.NUM_EDGE_TYPES,
    dropout=0.1
).to(device)

diffusion = GaussianDiffusion(
    model=denoise_model,
    image_size=cfg.IMAGE_SIZE,
    timesteps=cfg.TIMESTEPS,
    beta_start=cfg.BETA_START,
    beta_end=cfg.BETA_END
).to(device)

optimizer = torch.optim.Adam(list(denoise_model.parameters()) + list(graph_encoder.parameters()), lr=cfg.LEARNING_RATE)
scaler = torch.amp.GradScaler('cuda', enabled=True)

torch.backends.cudnn.benchmark = True

t0 = time.time()
denoise_model.train()
print("Benchmarking training loop...")
for i, batch in enumerate(train_loader):
    batch = batch.to(device)
    optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(device_type=device.type, enabled=True):
        global_emb, _ = graph_encoder(batch.x, batch.edge_index, batch.edge_attr if hasattr(batch, "edge_attr") else None)
        if global_emb.dim() == 2:
            if global_emb.shape[0] == 1 and batch.target_img.shape[0] > 1:
                global_emb = global_emb.repeat(batch.target_img.shape[0], 1)
        else:
            global_emb = global_emb.unsqueeze(0)
            if global_emb.shape[0] == 1 and batch.target_img.shape[0] > 1:
                global_emb = global_emb.repeat(batch.target_img.shape[0], 1)
        t = torch.randint(0, diffusion.timesteps, (batch.target_img.shape[0],), device=device).long()
        loss = diffusion.p_losses(batch.target_img, t, None, None, global_emb)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    if i == 50:
        break

t1 = time.time()
print(f"Time for 50 batches: {t1 - t0:.2f} seconds ({(t1 - t0) / 50:.2f} s/batch)")
