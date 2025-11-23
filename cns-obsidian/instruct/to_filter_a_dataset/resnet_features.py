import os
import pandas as pd
from PIL import Image
from collections.abc import Iterable


def get_resnet_features(image):
    import torch
    import torchvision.transforms as transforms
    import torchvision.models as models

    if "resnet" not in get_resnet_features.__dict__:
        # Instantiate ResNet model only if it's not already loaded
        weights = models.ResNet50_Weights.IMAGENET1K_V2
        get_resnet_features.resnet = models.resnet50(weights=weights)
        get_resnet_features.resnet.eval()

    if "transform" not in get_resnet_features.__dict__:
        get_resnet_features.transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(
                    224
                ),  # Assuming ResNet takes 224x224 images
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    if image.mode != "RGB":
        # Grayscale or CMYK image, convert it to RGB
        image = image.convert("RGB")

    image_tensor = get_resnet_features.transform(image).unsqueeze(0)
    with torch.no_grad():
        features = get_resnet_features.resnet(image_tensor)

    return features


# Factory function to create the filtering function
def create_filtering_function(df_path, image_root, positive_labels):
    from sklearn.linear_model import LogisticRegression

    # Load the dataframe
    df = pd.read_csv(df_path, index_col=0).sample(frac=1)

    # Handle non-iterable positive_labels
    if not isinstance(positive_labels, Iterable) or isinstance(
        positive_labels, str
    ):
        positive_labels = [positive_labels]

    # Prepare the data
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].apply(lambda x: x in positive_labels).values

    # Train the logistic regression model
    clf = LogisticRegression(random_state=0).fit(X, y)

    def filtering_function(entry):
        # Construct image path
        image_path = os.path.join(image_root, entry["image"])

        # Open image
        image = Image.open(image_path)

        # Extract features
        features = get_resnet_features(image)

        # Predict the label
        prediction = clf.predict(features)

        return bool(prediction[0])

    return filtering_function
