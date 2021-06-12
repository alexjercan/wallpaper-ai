from api.general import get_config, get_model, get_predictions, get_transform
from src.dataset import LoadURIs


def filter_uri(images):
    config = get_config()
    model = get_model(config)
    transform = get_transform(config)
    dataset = LoadURIs(images, transform)

    return list(get_predictions(model, dataset))
