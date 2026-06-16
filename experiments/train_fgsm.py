import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import resnet18, resnet34, resnet50
import torch.optim as optim
import copy
import time

# Load data
data = np.load("/home/atml_team032/robust_classifier/train.npz", allow_pickle = True)
images = torch.from_numpy(data["images"]).float() / 255.0
labels = torch.from_numpy(data["labels"]).long()

print("Dataset size:", len(images))
print("Image shape:", images.shape)
print("Label range:", labels.min().item(), "to", labels.max().item())

#  Train / val split 
NUM_CLASSES  = 9
VAL_FRAC     = 0.05
N            = len(images)
n_val        = int(N * VAL_FRAC)
n_train      = N - n_val
idx          = torch.randperm(N, generator=torch.Generator().manual_seed(42))
tr_idx, va_idx = idx[:n_train], idx[n_train:]

train_dataset = TensorDataset(images[tr_idx], labels[tr_idx])
val_dataset   = TensorDataset(images[va_idx], labels[va_idx])

train_loader  = DataLoader(train_dataset, batch_size=128, shuffle=True,  num_workers=4, pin_memory=True)
val_loader    = DataLoader(val_dataset,   batch_size=256, shuffle=False, num_workers=4, pin_memory=True)

# Model (exactly as in task_template.py) 
model = resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)

# Sanity check
model.eval()
with torch.no_grad():
    out = model(torch.randn(1, 3, 32, 32))
print("Output shape:", out.shape)  # must be (1, 9)

# Hyperparameters 
NUM_EPOCHS   = 100
LR           = 0.1
MOMENTUM     = 0.9
WEIGHT_DECAY = 5e-4
EPS          = 8  / 255.0    # L-inf budget
STEP_SIZE    = 2  / 255.0    # PGD step
PGD_STEPS    = 10            # PGD iterations during training
ADV_WEIGHT   = 0.5           # 0.5*clean + 0.5*adv  (matches eval metric)
EMA_DECAY    = 0.9995
LABEL_SMOOTH = 0.1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
model = model.to(device)

#  EMA shadow model 
ema_model = copy.deepcopy(model)
for p in ema_model.parameters():
    p.requires_grad_(False)

def update_ema(model, ema_model, decay=EMA_DECAY):
    with torch.no_grad():
        for p_ema, p in zip(ema_model.parameters(), model.parameters()):
            p_ema.data.mul_(decay).add_(p.data, alpha=1 - decay)

# PGD attack 

"""
def pgd_attack(model, x, y, eps=EPS, step_size=STEP_SIZE, num_steps=PGD_STEPS):
    model.eval()
    x_adv = x.detach() + torch.zeros_like(x).uniform_(-eps, eps)
    x_adv = x_adv.clamp(0.0, 1.0)
    for _ in range(num_steps):
        x_adv.requires_grad_(True)
        loss = nn.CrossEntropyLoss()(model(x_adv), y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        with torch.no_grad():
            x_adv = x_adv + step_size * grad.sign()
            x_adv = x + (x_adv - x).clamp(-eps, eps)
            x_adv = x_adv.clamp(0.0, 1.0)
    model.train()
    return x_adv.detach()
    
    """
    
    
# Now trying out FGSM attack on resnet 18

def fgsm_attack(model, x, y, eps=EPS):
    model.eval()
    x_adv = x.detach().clone()
    x_adv.requires_grad_(True)

    loss = nn.CrossEntropyLoss()(model(x_adv), y)
    grad = torch.autograd.grad(loss, x_adv)[0]

    with torch.no_grad():
        x_adv = x_adv + eps * grad.sign()
        x_adv = x_adv.clamp(0.0, 1.0)

    model.train()
    return x_adv.detach()
    
    

# Optimiser & scheduler 
optimizer = optim.SGD(model.parameters(), lr=LR, momentum=MOMENTUM,
                      weight_decay=WEIGHT_DECAY, nesterov=True)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTH)

# Evaluation helpers 
@torch.no_grad()
def clean_accuracy(model, loader):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        correct += (model(x).argmax(1) == y).sum().item()
        total   += y.size(0)
    return correct / total

def robust_accuracy(model, loader, num_steps=20):
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y  = x.to(device), y.to(device)
        #x_adv = pgd_attack(model, x, y, num_steps=num_steps)
        x_adv = fgsm_attack(model, x, y)
        with torch.no_grad():
            correct += (model(x_adv).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / total

#  Training loop 
best_score = 0.0
best_state = None

print(f"\n{'Epoch':>6} {'LR':>8} {'Loss':>8} {'Clean':>8} {'Robust':>8} {'Score':>8} {'Time':>6}")
print("─" * 60)

for epoch in range(1, NUM_EPOCHS + 1):
    t0 = time.time()
    model.train()
    total_loss, n_batches = 0.0, 0

    for x, y in train_loader:
        x, y  = x.to(device), y.to(device)
        #x_adv = pgd_attack(model, x, y)
        x_adv = fgsm_attack(model, x, y)

        model.train()
        optimizer.zero_grad()
        loss = (1 - ADV_WEIGHT) * criterion(model(x),     y) + \
                    ADV_WEIGHT  * criterion(model(x_adv), y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        update_ema(model, ema_model)

        total_loss += loss.item()
        n_batches  += 1

    scheduler.step()
    lr_now = optimizer.param_groups[0]["lr"]

    # Evaluate EMA model
    clean_acc  = clean_accuracy(ema_model, val_loader)
    robust_acc = robust_accuracy(ema_model, val_loader, num_steps=10 if epoch % 10 != 0 else 20)
    score      = 0.5 * clean_acc + 0.5 * robust_acc

    print(f"{epoch:>6} {lr_now:>8.5f} {total_loss/n_batches:>8.4f} "
          f"{clean_acc:>7.3%} {robust_acc:>8.3%} {score:>8.3%} {time.time()-t0:>5.1f}s")

    if score > best_score and clean_acc > 0.50:
        best_score = score
        best_state = copy.deepcopy(ema_model.state_dict())
        torch.save(best_state, "model.pt")
        print(f" Saved (score={best_score:.3%})")

#  Final 
if best_state is None:
    torch.save(ema_model.state_dict(), "model.pt")

print(f"\nDone. Best score: {best_score:.3%}  →  model.pt")

# Sanity check
model_check = resnet18(weights=None)
model_check.fc = nn.Linear(model_check.fc.in_features, NUM_CLASSES)
model_check.load_state_dict(torch.load("model.pt", map_location="cpu"))
model_check.eval()
with torch.no_grad():
    out = model_check(torch.randn(1, 3, 32, 32))
assert out.shape == (1, 9), f"Bad shape: {out.shape}"
print("Sanity check passed Output shape:", out.shape)
