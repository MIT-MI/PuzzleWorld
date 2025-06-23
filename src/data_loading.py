import base64
import io
import json
from pathlib import Path
from typing import List, Tuple
import re

import requests
from PIL import Image
from fire import Fire
from pydantic import BaseModel

Point = Tuple[float, float]

DATA_DIR = Path("data")

def convert_image_to_text(image: Image) -> str:
    # This is also how OpenAI encodes images: https://platform.openai.com/docs/guides/vision
    with io.BytesIO() as output:
        image.save(output, format="PNG")
        data = output.getvalue()
    return base64.b64encode(data).decode("utf-8")


def convert_image_to_bytes(image: Image) -> bytes:
    with io.BytesIO() as output:
        image.save(output, format=image.format)
        data = output.getvalue()
    return data


def convert_text_to_image(text: str) -> Image:
    data = base64.b64decode(text.encode("utf-8"))
    return Image.open(io.BytesIO(data))


def load_image(path: str) -> Image:
    if Path(path).exists():
        return Image.open(path)

    response = requests.get(path)
    return Image.open(io.BytesIO(response.content))

class Step(BaseModel):
    explanation: str
    figure: str | None = None

class Sample(BaseModel):
    title: str
    flavor_text: str
    puzzle_content: List = []
    solution: str
    reasoning: List[Step]
    modality: List[str]
    skills: List[str]
    prompt: str = ""
    raw_output: str = ""


class Data(BaseModel):
    sample: Sample

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            print(self.sample.json(indent=2, exclude={"puzzle_content"}), file=f)
                
    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            sample = Sample(**json.loads(f.read()))
        return cls(sample=sample)
        with open(path, "r", encoding="utf-8") as f:
            samples = [Sample(**json.loads(f.read()))]
        print(dict(path=path, samples=len(samples)))
        return cls(samples=samples)

    @classmethod
    def loadlines(cls, path: str):
        sample: Sample
        with open(path) as f:
            for line in f:
                sample = Sample(**json.loads(line))
        return cls(samples=sample)

    @classmethod
    def load_with_puzzle_content(cls, path: str):
        data = cls.load(Path(path, "metadata.json"))

        path_obj = Path(path)
        # Find all files matching content*.png
        content_files = sorted(
            path_obj.glob("content*.png"),
            key=lambda f: (0 if f.name == "content.png" else int(re.search(r'\d+', f.stem).group() or 0))
        )
        
        images = [Image.open(f) for f in content_files]
        data.sample.puzzle_content = images
        return data


def test_data(**kwargs):
    data = Data.load_with_image_dir(**kwargs)
    data.analyze()


if __name__ == "__main__":
    Fire()
