# coding: utf-8
import numpy as np
import matplotlib.pyplot as plt
from load_mnist import load_mnist
from util import smooth_curve
from multi_layer_net import MultiLayerNet
from optimizer import *

# 0: Read the MNIST data
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
train_size = x_train.shape[0]
batch_size = 128
max_iterations = 1000

# Policy 1: Default learning rates (baseline)
learning_rates = {
    'SGD': 0.01,
    'Momentum': 0.01,
    'AdaGrad': 0.01,
    'Adam': 0.001
}

# Policy 2: High learning rates (uncomment to test)
# learning_rates = {
#     'SGD': 0.1,
#     'Momentum': 0.1,
#     'AdaGrad': 0.1,
#     'Adam': 0.01
# }

# Policy 3: Low learning rates (uncomment to test)
# learning_rates = {
#     'SGD': 0.001,
#     'Momentum': 0.001,
#     'AdaGrad': 0.001,
#     'Adam': 0.0001
# }

# Policy 4: Will use step decay during training (see below)
use_step_decay = False  # Set to True to enable step decay
step_decay_milestones = [300, 600, 900]
step_decay_rate = 0.5

# Policy 5: Will use exponential decay during training (see below)
use_exponential_decay = False # Set to True to enable exponential decay
exponential_decay_rate = 0.95
exponential_decay_interval = 100

# 1: Set the experiment
optimizers = {}
optimizers['SGD'] = SGD(lr=learning_rates['SGD'])
optimizers['Momentum'] = Momentum(lr=learning_rates['Momentum'])
optimizers['AdaGrad'] = AdaGrad(lr=learning_rates['AdaGrad'])
optimizers['Adam'] = Adam(lr=learning_rates['Adam'])

networks = {}
train_loss = {}
test_acc = {}      # For Question 2
train_acc = {}     # For Question 2
learning_rates_history = {key: [] for key in optimizers.keys()}  # Track LR changes

for key in optimizers.keys():
    networks[key] = MultiLayerNet(
        input_size=784, hidden_size_list=[100, 100, 100, 100],
        output_size=10)
    train_loss[key] = []
    test_acc[key] = []
    train_acc[key] = []

# 2: Start training
eval_interval = 100
initial_lr = learning_rates.copy()  # Store initial LR for decay policies

for i in range(max_iterations):
    # ========== Learning Rate Policy Implementation ==========
    # Step Decay (Policy 4)
    if use_step_decay and i in step_decay_milestones:
        for key in optimizers.keys():
            optimizers[key].lr *= step_decay_rate
        print(f"\n[Step Decay] Iteration {i}: Learning rates reduced by {step_decay_rate}x\n")
    
    # Exponential Decay (Policy 5)
    if use_exponential_decay and i % exponential_decay_interval == 0 and i > 0:
        for key in optimizers.keys():
            optimizers[key].lr = initial_lr[key] * (exponential_decay_rate ** (i / exponential_decay_interval))
    
    # Get batch
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]
    
    # Training step
    for key in optimizers.keys():
        grads = networks[key].gradient(x_batch, t_batch)
        optimizers[key].update(networks[key].params, grads)
        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)
        learning_rates_history[key].append(optimizers[key].lr)
    
    if i % eval_interval == 0:
        print("===========" + " iteration:" + str(i) + " ===========")
        for key in optimizers.keys():
            loss = networks[key].loss(x_batch, t_batch)
            train_accuracy = networks[key].accuracy(x_train, t_train)
            test_accuracy = networks[key].accuracy(x_test, t_test)
            
            train_acc[key].append(train_accuracy)
            test_acc[key].append(test_accuracy)
            
            print(f"{key:10s}: loss={loss:.4f}, train_acc={train_accuracy:.4f}, test_acc={test_accuracy:.4f}, lr={optimizers[key].lr:.6f}")

# 3: Draw graphs
markers = {"SGD": "o", "Momentum": "x", "AdaGrad": "s", "Adam": "D"}
x_loss = np.arange(max_iterations)
x_acc = np.arange(0, max_iterations, eval_interval)

# Graph 1: Training Loss Comparison (Question 1)
plt.figure(figsize=(10, 6))
for key in optimizers.keys():
    plt.plot(x_loss, smooth_curve(train_loss[key]), 
             marker=markers[key], 
             markevery=100, 
             label=key)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.ylim(0, 1)
plt.title("Training Loss Comparison Across Optimizers")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Graph 2: Test Accuracy Over Time (Question 2)
plt.figure(figsize=(10, 6))
for key in optimizers.keys():
    plt.plot(x_acc, test_acc[key], 
             marker=markers[key], 
             markevery=1, 
             label=key)
plt.xlabel("Iterations")
plt.ylabel("Test Accuracy")
plt.ylim(0, 1)
plt.title("Test Accuracy Comparison Across Optimizers")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Graph 3: Train vs Test Accuracy (Question 2 - Overfitting Detection)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, key in enumerate(optimizers.keys()):
    axes[idx].plot(x_acc, train_acc[key], marker='o', markersize=4, label='Train Acc', linewidth=2)
    axes[idx].plot(x_acc, test_acc[key], marker='s', markersize=4, label='Test Acc', linewidth=2)
    axes[idx].set_xlabel("Iterations")
    axes[idx].set_ylabel("Accuracy")
    axes[idx].set_ylim(0, 1)
    axes[idx].set_title(f"{key} - Train vs Test Accuracy")
    axes[idx].legend()
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Graph 4: Learning Rate History (Optional - for decay policies)
if use_step_decay or use_exponential_decay:
    plt.figure(figsize=(10, 6))
    for key in optimizers.keys():
        plt.plot(x_loss, learning_rates_history[key], label=key, linewidth=2)
    plt.xlabel("Iterations")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedule Over Time")
    plt.legend()
    plt.grid(True)
    plt.yscale('log')  # Log scale to see decay clearly
    plt.tight_layout()
    plt.show()

print("\n========== Training Complete ==========")
print("Final Results:")
for key in optimizers.keys():
    print(f"{key:10s}: Final Test Accuracy = {test_acc[key][-1]:.4f}")