from src.model import Model
from src.config import DEVICE, parse_detect_config, read_yaml_config
from src.general import load_checkpoint
import torch
import albumentations as A
import src.my_albumentations as M


def get_config():
    return parse_detect_config(read_yaml_config("detect.yaml"))


def get_model(config):
    model = Model()
    model = model.to(DEVICE)
    _, model = load_checkpoint(model, "../normal.pth", DEVICE)
    return model


def get_transform(config):
    return A.Compose(
        [
            A.Resize(height=config.IMAGE_SIZE[0], width=config.IMAGE_SIZE[1]),
            A.Normalize(),
            M.MyToTensorV2(),
        ],
    )


def get_predictions(model, dataset):
    for og_img, img, path in dataset:
        with torch.no_grad():
            img = img.to(DEVICE, non_blocking=True).unsqueeze(0)

            predictions = model(img)
            yield og_img, predictions, path


def build_response(predictions):
    arr = []
    for _, pred, img in predictions:
        print(pred)
        _, pred = torch.max(pred, 1)
        arr.append({ "image": img, "label": pred.item() })
    return arr