import torch

def _succ(P):
    return torch.roll(P, -1, dims=0)

def ortho_loss(P):
    Q = _succ(P)
    edge = (Q - P).abs()
    length = torch.sqrt((edge * edge).sum(-1) + 1e-6)
    ortho = torch.minimum(edge[..., 0], edge[..., 1]) / (length + 1e-3)
    return (ortho * length).mean()

# Triangle-like shape
P = torch.tensor([[0.0, 0.0], [10.0, 2.0], [5.0, 10.0]], requires_grad=True)
opt = torch.optim.Adam([P], lr=0.5)

for i in range(200):
    opt.zero_grad()
    loss = ortho_loss(P)
    loss.backward()
    opt.step()

print("After:", P.detach().numpy())
print("Loss:", ortho_loss(P).item())
