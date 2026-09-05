from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from transformers import (
    AutoModel,
    AutoProcessor,
)


@dataclass
class ImageEmbedding:
    tile_id: str
    image_path: str
    vector: list[float]


class SigLIPEncoder:

    def __init__(
        self,
        model_name: str = "google/siglip-base-patch16-224",
    ):
        self.model_name = model_name

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"Loading visual encoder: {model_name}"
        )
        print(
            f"Device: {self.device}"
        )

        self.processor = AutoProcessor.from_pretrained(
            model_name
        )

        self.model = AutoModel.from_pretrained(
            model_name
        )

        self.model.to(self.device)
        self.model.eval()

    def encode_image(
        self,
        image_path: Path,
    ) -> list[float]:

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        with Image.open(image_path) as image:

            image = image.convert("RGB")

            inputs = self.processor(
                images=image,
                return_tensors="pt",
            )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = self.model.get_image_features(
                **inputs
            )

        embedding = outputs.pooler_output

        embedding = embedding.squeeze(0)

        embedding = embedding / embedding.norm(
            p=2
        )

        return embedding.cpu().tolist()

    def encode_text(
        self,
        text: str,
    ) -> list[float]:

        if not text.strip():
            raise ValueError(
                "Text query cannot be empty."
            )

        inputs = self.processor(
            text=[text],
            padding="max_length",
            return_tensors="pt",
        )

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        with torch.no_grad():

            outputs = self.model.get_text_features(
                **inputs
            )

        embedding = outputs.pooler_output

        embedding = embedding.squeeze(0)

        embedding = embedding / embedding.norm(
            p=2
        )

        return embedding.cpu().tolist()