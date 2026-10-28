import time, os
import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision import datasets, transforms
from PIL import Image, ImageOps

# ---------------- Hiperparametreler ----------------
SEED = 42
BATCH_SIZE = 128
EPOCHS = 20
LR = 1e-3
HIDDEN = [256, 128]       # 2 hidden layer
DROPOUT = 0.2
OPTIMIZER = "Adam"
VAL_SIZE = 10000          # 60000 -> 50000 train + 10000 val

torch.manual_seed(SEED); np.random.seed(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# ---------------- Veri ----------------
MEAN, STD = 0.1307, 0.3081
tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((MEAN,), (STD,))])

full_train = datasets.MNIST("./data", train=True, download=True, transform=tf)
test_set   = datasets.MNIST("./data", train=False, download=True, transform=tf)
train_set, val_set = random_split(
    full_train, [len(full_train) - VAL_SIZE, VAL_SIZE],
    generator=torch.Generator().manual_seed(SEED))

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=1000)
test_loader  = DataLoader(test_set,  batch_size=1000)
print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

# ---------------- Model ----------------
class MLP(nn.Module):
    def __init__(self, hidden=HIDDEN, dropout=DROPOUT):
        super().__init__()
        layers, in_dim = [nn.Flatten()], 28 * 28
        for h in hidden:
            layers += [nn.Linear(in_dim, h), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = h
        layers.append(nn.Linear(in_dim, 10))   # logits (CrossEntropyLoss softmax'ı içerir)
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

model = MLP().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

def run_epoch(loader, train=False):
    model.train(train)
    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            if train:
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            total_loss += loss.item() * x.size(0)
            correct += (out.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n

# ---------------- Eğitim ----------------
hist = {"tr_loss": [], "va_loss": [], "tr_acc": [], "va_acc": []}
start = time.time()
for ep in range(1, EPOCHS + 1):
    tl, ta = run_epoch(train_loader, True)
    vl, va = run_epoch(val_loader, False)
    for k, v in zip(hist, [tl, vl, ta, va]): hist[k].append(v)
    print(f"Epoch {ep:02d} | train loss {tl:.4f} acc {ta:.4f} | val loss {vl:.4f} acc {va:.4f}")
train_time = time.time() - start

# Test seti SADECE en sonda, bir kez kullanılır
test_loss, test_acc = run_epoch(test_loader, False)

print("\n===== SONUÇLAR =====")
print(f"Final training loss      : {hist['tr_loss'][-1]:.4f}")
print(f"Final validation loss    : {hist['va_loss'][-1]:.4f}")
print(f"Final training accuracy  : {hist['tr_acc'][-1]*100:.2f}%")
print(f"Final validation accuracy: {hist['va_acc'][-1]*100:.2f}%")
print(f"Test accuracy            : {test_acc*100:.2f}%")
print(f"Eğitim süresi            : {train_time:.1f} sn")

# ---------------- Grafikler ----------------
ep_range = range(1, EPOCHS + 1)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot(ep_range, hist["tr_loss"], label="Train"); ax[0].plot(ep_range, hist["va_loss"], label="Validation")
ax[0].set_title("Loss"); ax[0].set_xlabel("Epoch"); ax[0].legend(); ax[0].grid(alpha=.3)
ax[1].plot(ep_range, hist["tr_acc"], label="Train"); ax[1].plot(ep_range, hist["va_acc"], label="Validation")
ax[1].set_title("Accuracy"); ax[1].set_xlabel("Epoch"); ax[1].legend(); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig("training_curves.png", dpi=150); plt.show()

torch.save(model.state_dict(), "mlp_mnist.pt")


def preprocess(path):
    from PIL import ImageFilter
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32)

    # Koyu zemin / açık rakam ise ters çevir; hedef: koyu mürekkep, açık kağıt
    if arr.mean() < 127:
        arr = 255 - arr

    # Kağıt ve mürekkep seviyesini bul, gölgeyi/gürültüyü temizle
    bg = np.percentile(arr, 90)            # kağıt parlaklığı
    ink = np.percentile(arr, 1)            # mürekkep parlaklığı
    ink_map = np.clip((bg - arr) / (bg - ink + 1e-6), 0, 1)
    ink_map[ink_map < 0.35] = 0            # silik gölgeleri sil
    ink_map = (ink_map * 255).astype(np.uint8)

    img = Image.fromarray(ink_map)         # siyah zemin, beyaz rakam (MNIST formatı)
    img = img.filter(ImageFilter.MaxFilter(5))   # çizgiyi kalınlaştır

    # Rakamın etrafını kırp, kareye tamamla, 20x20'ye sığdır, 28x28'e ortala
    a = np.array(img)
    ys, xs = np.where(a > 50)
    if len(ys) > 0:
        img = img.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    w, h = img.size; s = max(w, h)
    canvas = Image.new("L", (s, s), 0)
    canvas.paste(img, ((s - w) // 2, (s - h) // 2))
    canvas = canvas.resize((20, 20), Image.LANCZOS)
    final = Image.new("L", (28, 28), 0)
    final.paste(canvas, (4, 4))

    x = np.array(final, dtype=np.float32) / 255.0
    x = (x - MEAN) / STD
    return final, torch.tensor(x).unsqueeze(0).unsqueeze(0)


folder = "my_digits"
if os.path.isdir(folder):
    files = sorted(f for f in os.listdir(folder) if f.lower().endswith((".png", ".jpg", ".jpeg")))
    model.eval(); correct = 0
    fig, axes = plt.subplots(1, len(files), figsize=(2 * len(files), 2.6))
    axes = np.atleast_1d(axes)
    for a, f in zip(axes, files):
        true = int(f[0])
        img28, x = preprocess(os.path.join(folder, f))
        with torch.no_grad():
            probs = torch.softmax(model(x.to(device)), 1)[0]
        pred = probs.argmax().item(); correct += (pred == true)
        a.imshow(img28, cmap="gray"); a.axis("off")
        a.set_title(f"T:{true} P:{pred}\n{probs[pred]*100:.0f}%", color="g" if pred == true else "r")
        print(f"{f}: gerçek={true}, tahmin={pred}, güven={probs[pred]*100:.1f}%")
    print(f"\nKendi el yazısı doğruluğu: {correct}/{len(files)} = {correct/len(files)*100:.1f}%")
    plt.tight_layout(); plt.savefig("my_digits_predictions.png", dpi=150); plt.show()
else:
    print("my_digits/ klasörü bulunamadı; el yazısı kısmı atlandı.")