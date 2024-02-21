import torch

INPUT_IMAGE_HEIGHT = 500
INPUT_IMAGE_WIDTH = 500
BATCH_SIZE = 60
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PIN_MEMORY = True if DEVICE == "cuda" else False