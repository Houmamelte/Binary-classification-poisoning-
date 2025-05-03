import torch
import torch.nn.utils as utils
from sklearn.metrics import accuracy_score

class Trainer:
    def __init__(self, model, criterion, optimizer, scheduler, device):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device).float()
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item() * inputs.size(0)
        return total_loss / len(dataloader.dataset)

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        preds, labels = [], []
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs, targets = inputs.to(self.device), targets.to(self.device).float()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item() * inputs.size(0)
                pred_labels = torch.round(torch.sigmoid(outputs))
                preds.extend(pred_labels.cpu().numpy())
                labels.extend(targets.cpu().numpy())
        acc = accuracy_score(labels, preds)
        return total_loss / len(dataloader.dataset), acc
