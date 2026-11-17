"""BigQuant submission entry; shared implementation lives in train.py/predict.py."""
from predict import main
from train import train_and_save

__all__ = ['main', 'train_and_save']
