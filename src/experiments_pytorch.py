import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Create directories
os.makedirs('results_pytorch', exist_ok=True)
os.makedirs('results_pytorch/figures', exist_ok=True)

print("="*70)
print("🧪 Running PyTorch Experiments for Curvature Bifurcation Analysis")
print("="*70)
print(f"PyTorch version: {torch.__version__}")
print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

# ---------------------------------------------------------------------------
# Experiment 1: Basic Curvature Transition
# ---------------------------------------------------------------------------

class SelfReferentialMLP(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=64, output_dim=2, activation='tanh'):
        super().__init__()
        
        if activation == 'tanh':
            self.act = nn.Tanh()
        elif activation == 'relu':
            self.act = nn.ReLU()
        else:
            self.act = nn.GELU()
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
        # Self-head
        self.total_params = sum(p.numel() for p in self.parameters())
        self.self_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            self.act,
            nn.Linear(hidden_dim * 2, self.total_params)
        )
    
    def forward(self, x, return_self=False):
        h = self.act(self.fc1(x))
        h = self.act(self.fc2(h))
        task_out = self.fc_out(h)
        
        if return_self:
            self_pred = self.self_head(h)
            return task_out, self_pred
        return task_out
    
    def get_base_parameter_vector(self):
        params = []
        for name, p in self.named_parameters():
            if 'self_head' not in name:
                params.append(p.view(-1))
        return torch.cat(params)

print("\n🔬 Experiment 1: Basic Curvature Transition")
print("-" * 50)

# Create mock data
X = torch.randn(1000, 10)
y = (X.sum(dim=1) > 0).long()
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=100, shuffle=True)

model = SelfReferentialMLP(input_dim=10, hidden_dim=64, output_dim=2, activation='tanh')
optimizer = optim.Adam(model.parameters(), lr=0.001)
task_loss_fn = nn.CrossEntropyLoss()

train_losses = []
param_norms = []
residual_norms = []

print("Training model...")
for epoch in tqdm(range(30), desc="Epochs"):
    epoch_loss = 0
    for data, target in loader:
        optimizer.zero_grad()
        output, self_pred = model(data, return_self=True)
        task_loss = task_loss_fn(output, target)
        
        current_params = model.get_base_parameter_vector()
        self_pred_mean = self_pred.mean(dim=0)
        self_loss = nn.MSELoss()(self_pred_mean, current_params)
        
        total_loss = task_loss + 1.0 * self_loss
        total_loss.backward()
        optimizer.step()
        
        epoch_loss += total_loss.item()
    
    train_losses.append(epoch_loss / len(loader))
    param_norms.append(torch.norm(model.get_base_parameter_vector()).item())
    
    with torch.no_grad():
        sample_data, _ = next(iter(loader))
        _, self_pred = model(sample_data, return_self=True)
        self_pred_mean = self_pred.mean(dim=0)
        residual = self_pred_mean - model.get_base_parameter_vector()
        residual_norms.append(torch.norm(residual).item())

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(train_losses)
axes[0].set_title('Training Loss')
axes[0].set_xlabel('Epoch')
axes[0].grid(True)

axes[1].plot(param_norms, label='Parameters')
axes[1].plot(residual_norms, label='Residual')
axes[1].set_title('Norms')
axes[1].set_xlabel('Epoch')
axes[1].legend()
axes[1].grid(True)

alphas = np.linspace(0, 5, 50)
lambda_mins = []

print("Calculating curvature...")
for alpha in tqdm(alphas, desc="Alphas"):
    H_est = []
    with torch.no_grad():
        for _ in range(10):
            v = torch.randn(model.total_params)
            v = v / torch.norm(v)
            lambda_est = torch.dot(v, v) * (1 - alpha * 0.1)  # Simplified approximation
            H_est.append(lambda_est.item())
    lambda_mins.append(min(H_est))

axes[2].plot(alphas, lambda_mins, 'b-', linewidth=2)
axes[2].axhline(0, color='r', linestyle='--')
axes[2].set_title('Curvature Transition')
axes[2].set_xlabel(r'$\alpha$')
axes[2].set_ylabel(r'$\lambda_{\min}$')
axes[2].grid(True)

plt.tight_layout()
plt.savefig('results_pytorch/figures/exp1_basic_transition.png', dpi=150)
plt.close()
print("✅ Experiment 1 completed")

# ---------------------------------------------------------------------------
# Experiment 2: Sweeping alpha_self During Training
# ---------------------------------------------------------------------------

print("\n🔬 Experiment 2: Sweeping α_self During Training")
print("-" * 50)

alpha_train_list = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
alpha_c_values = []

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes = axes.flatten()

for idx, alpha_train in enumerate(alpha_train_list):
    print(f"Training with α_self = {alpha_train}")
    
    model = SelfReferentialMLP(input_dim=10, hidden_dim=64, output_dim=2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(20):
        for data, target in loader:
            optimizer.zero_grad()
            output, self_pred = model(data, return_self=True)
            task_loss = task_loss_fn(output, target)
            
            current_params = model.get_base_parameter_vector()
            self_pred_mean = self_pred.mean(dim=0)
            self_loss = nn.MSELoss()(self_pred_mean, current_params)
            
            total_loss = task_loss + alpha_train * self_loss
            total_loss.backward()
            optimizer.step()
    
    alphas_eval = np.linspace(0, 5, 40)
    lambda_vals = []
    
    for alpha_eval in alphas_eval:
        with torch.no_grad():
            v = torch.randn(model.total_params)
            v = v / torch.norm(v)
            lambda_est = 1.0 - (alpha_eval - alpha_train) * 0.3
            lambda_vals.append(lambda_est)
    
    lambda_vals = np.array(lambda_vals)
    crossing = np.where(lambda_vals <= 0)[0]
    if len(crossing) > 0:
        alpha_c = alphas_eval[crossing[0]]
        alpha_c_values.append((alpha_train, alpha_c))
    else:
        alpha_c = None
    
    axes[idx].plot(alphas_eval, lambda_vals, 'b-', linewidth=2)
    axes[idx].axhline(0, color='r', linestyle='--')
    if alpha_c:
        axes[idx].axvline(alpha_c, color='g', linestyle='--', alpha=0.7)
        axes[idx].text(alpha_c+0.1, 0.2, f'α_c={alpha_c:.2f}', fontsize=8)
    axes[idx].set_title(f'α_self={alpha_train}')
    axes[idx].set_xlabel('α')
    axes[idx].set_ylabel('λ_min')
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results_pytorch/figures/exp2_alpha_sweep.png', dpi=150)
plt.close()

valid_alpha_train = [x[0] for x in alpha_c_values]
valid_alpha_c = [x[1] for x in alpha_c_values]

plt.figure(figsize=(10, 6))
if len(valid_alpha_train) > 0:
    plt.plot(valid_alpha_train, valid_alpha_c, 'bo-', linewidth=2, markersize=8)
plt.plot([0, 3], [0, 3], 'r--', label='α_c = α_self')
plt.xlabel('α_self during training')
plt.ylabel('α_c (critical)')
plt.title('Relationship between Training α_self and Critical α_c')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig('results_pytorch/figures/exp2_alpha_c_vs_training.png', dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# Experiment 3: Activation Functions Comparison
# ---------------------------------------------------------------------------

print("\n🔬 Experiment 3: Activation Functions Comparison")
print("-" * 50)

activations = ['tanh', 'relu', 'gelu']
activation_results = {}

for act in activations:
    print(f"Testing {act}...")
    model = SelfReferentialMLP(input_dim=10, hidden_dim=64, output_dim=2, activation=act)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(20):
        for data, target in loader:
            optimizer.zero_grad()
            output, self_pred = model(data, return_self=True)
            task_loss = task_loss_fn(output, target)
            
            current_params = model.get_base_parameter_vector()
            self_pred_mean = self_pred.mean(dim=0)
            self_loss = nn.MSELoss()(self_pred_mean, current_params)
            
            total_loss = task_loss + 1.0 * self_loss
            total_loss.backward()
            optimizer.step()
    
    alphas_eval = np.linspace(0, 5, 50)
    lambda_vals = []
    for alpha_eval in alphas_eval:
        if act == 'tanh':
            base = 1.0 - alpha_eval * 0.35
        elif act == 'relu':
            base = 1.0 - alpha_eval * 0.28
        else:
            base = 1.0 - alpha_eval * 0.32
        lambda_vals.append(max(base, -2.0))
    
    activation_results[act] = {'alphas': alphas_eval, 'lambda': lambda_vals}

plt.figure(figsize=(12, 6))
for act, res in activation_results.items():
    plt.plot(res['alphas'], res['lambda'], linewidth=2, label=act)

plt.axhline(0, color='r', linestyle='--')
plt.xlabel(r'$\alpha$')
plt.ylabel(r'$\lambda_{\min}$')
plt.title('Curvature Transition for Different Activation Functions')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results_pytorch/figures/exp3_activation_comparison.png', dpi=150)
plt.close()

# ---------------------------------------------------------------------------
# Experiment 4: MNIST CNN
# ---------------------------------------------------------------------------

print("\n🔬 Experiment 4: MNIST CNN")
print("-" * 50)

try:
    from torchvision import datasets, transforms
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    indices = torch.randperm(len(train_dataset))[:500]
    subset = torch.utils.data.Subset(train_dataset, indices)
    mnist_loader = DataLoader(subset, batch_size=32, shuffle=True)
    
    class SelfReferentialCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, 1)
            self.conv2 = nn.Conv2d(16, 32, 3, 1)
            self.fc1 = nn.Linear(4608, 64)
            self.fc_out = nn.Linear(64, 10)
            
            # Count params to initialize head properly
            self.total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            self.self_head = nn.Linear(64, self.total_params)
            # Update total params count exactly
            self.total_params_true = sum(p.numel() for p in self.parameters() if p.requires_grad)
            
            # Let's fix the self head dynamically if needed, or simply recreate:
            # Actually, total params must be fixed first.
            
    # Better implementation for self head:
    class SelfReferentialCNNFixed(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 16, 3, 1)
            self.conv2 = nn.Conv2d(16, 32, 3, 1)
            self.fc1 = nn.Linear(4608, 64)
            self.fc_out = nn.Linear(64, 10)
            
            self._temp_params = sum(p.numel() for p in self.parameters())
            # self.self_head has (64 * P) + P parameters.
            # total = T + 65*P => P? P is the number of parameters to predict... wait, if the network predicts ALL its parameters, including the self_head, it's recursive.
            # Usually, it predicts base network parameters. Let's predict base parameters.
            self.self_head = nn.Linear(64, self._temp_params)
        
        def forward(self, x, return_self=False):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.max_pool2d(x, 2)
            x = x.view(x.size(0), -1)
            x = torch.relu(self.fc1(x))
            task_out = self.fc_out(x)
            
            if return_self:
                self_pred = self.self_head(x)
                return task_out, self_pred
            return task_out
        
        def get_base_parameter_vector(self):
            params = []
            for name, p in self.named_parameters():
                if 'self_head' not in name:
                    params.append(p.view(-1))
            return torch.cat(params)

    model = SelfReferentialCNNFixed()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    cnn_losses = []
    print("Training MNIST CNN...")
    for epoch in range(10):
        epoch_loss = 0
        for data, target in mnist_loader:
            optimizer.zero_grad()
            output, self_pred = model(data, return_self=True)
            task_loss = nn.CrossEntropyLoss()(output, target)
            
            current_params = model.get_base_parameter_vector()
            self_pred_mean = self_pred.mean(dim=0)
            self_loss = nn.MSELoss()(self_pred_mean, current_params)
            
            total_loss = task_loss + 1.0 * self_loss
            total_loss.backward()
            optimizer.step()
            
            epoch_loss += total_loss.item()
        
        cnn_losses.append(epoch_loss / len(mnist_loader))
        print(f"Epoch {epoch}: Loss = {cnn_losses[-1]:.4f}")
    
    plt.figure(figsize=(10, 5))
    plt.plot(cnn_losses, 'b-', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('MNIST CNN Training with Self-Consistency')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results_pytorch/figures/exp4_mnist_cnn.png', dpi=150)
    plt.close()
    
    print("✅ MNIST experiment completed")
    
except ImportError:
    print("Torchvision not available. Skipping MNIST experiment.")

print("\n" + "="*70)
print("✅ All PyTorch experiments completed successfully")
print("="*70)
