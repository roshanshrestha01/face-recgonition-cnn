from dataloaders import train_dataloader, validate_dataloader, test_dataloader
from settings import SHOW_BATCHES
from utils import show_batch

if SHOW_BATCHES['train']:
    for X, y in train_dataloader:
        show_batch(X, y)

if SHOW_BATCHES['validate']:
    for X, y in validate_dataloader:
        show_batch(X, y)

if SHOW_BATCHES['test']:
    for X, y in test_dataloader:
        show_batch(X, y)
