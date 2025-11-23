import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from datasets import CTUSDataset
from ThyroidFusion_skip_HAPE_PE_LSF_AGFusion_ADAACE import ThyroidFusionGated as SiameseNetwork
from AdaptiveACE import AdaptiveACE
from tqdm import tqdm
#from Loss.ACE_Loss import ACE_Loss
# Set random seed for reproducibility
torch.manual_seed(666)
save_pth = 'skip_HAPE_PE_LSF_AGFusion_ADAACELOSS.pth'
loss_log_path = 'loss_log_skip_HAPE_PE_LSF_AGFusion_ADAACELOSS.txt'  # 新增：损失记录文件的路径

# Hyperparameters`
num_classes = 3
learning_rate = 0.0001
batch_size = 16
num_epochs = 300
eval_every = 1  # Evaluate on validation set every eval_every epochs

# Check if GPU is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define transformations for image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load your dataset
train_dataset = CTUSDataset(root_dir='../data/train', transform=transform)
val_dataset = CTUSDataset(root_dir='../data/val', transform=transform)


# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

# Initialize your model
model = SiameseNetwork(num_classes).to(device)

# Define loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# criterion = nn.NLLLoss()
#loss_fn = nn.CrossEntropyLoss()
loss_fn = AdaptiveACE()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.2, verbose=True)
best_val_accuracy = 0.0  # Initialize best validation accuracy

# Function to evaluate model on test set
def evaluate(model, data_loader):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for ct_image, us_image, labels in data_loader:
            ct_image, us_image, labels = ct_image.to(device), us_image.to(device), labels.to(device)
            outputs, ct_f, us_f = model(ct_image, us_image)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    return accuracy

# Training loop
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    # Create progress bar
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch')

    for i, (ct_image, us_image, labels) in progress_bar:
        # Move tensors to the configured device
        ct_image, us_image, labels = ct_image.to(device), us_image.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs, ct_f, us_f = model(ct_image, us_image)
        loss = loss_fn(outputs, labels, ct_f, us_f)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Compute accuracy
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Print statistics
        running_loss += loss.item()
        progress_bar.set_postfix({'Loss': running_loss / (i + 1), 'Accuracy': (correct / total) * 100})
    # 新增：将平均损失写入到文件中
    with open(loss_log_path, 'a') as f:
        f.write(f'Epoch {epoch + 1}: Loss = {running_loss / len(train_loader)}\n')
    # Evaluate model on validation set every eval_every epochs
    if (epoch + 1) % eval_every == 0:
        val_accuracy = evaluate(model, val_loader)
        print(f'Validation Accuracy after {epoch + 1} epochs: {val_accuracy:.2f}%, Best acc is :{best_val_accuracy:.2f}%')

        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), save_pth)
            print('Best model saved')
        scheduler.step(val_accuracy)

print('Finished Training')
