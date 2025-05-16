import os
import torch
from PIL import Image
from dotenv import load_dotenv
from openai import OpenAI
from typing import Optional, List
from data_loading import convert_image_to_text
# from qwen_vl_utils import process_vision_info
from pathlib import Path
from transformers import (
    AutoProcessor,
    AutoModelForCausalLM,
    AutoModel,
    Qwen2VLForConditionalGeneration, 
    AutoTokenizer,
)
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

PUZZLE_SYSTEM_PROMPT = """
You will be presented with a puzzle to solve. The puzzle may not have specific instructions,
but you know that the answer to the puzzle is a word or short phrase (or rarely, a number).

Do not ask any questions about how to proceed, just do your best to solve the puzzle.
Here are some tips for solving puzzles of this type:

General Tips:
- Puzzles will often have multiple steps to get to the answer word. You can usually tell you
are on the right track if the intermediate answers agree with the title, flavor, or theme
of the puzzle.
- You can usually find hints in the introductory text. For example references to "in the dark"
or "sight" are often hints something is encoded with braille.
- Puzzles often incorporate acrostics: a clue where the first letter, syllable, or word of
each line, paragraph, or other recurring feature spells out a word or message.
- If you end up with a garbled "alphabet soup", then look for a clue on how to order them.
- Indexing is one of the most common puzzle mechanisms. Try indexing when you have a list of
words or phrases and a corresponding list of numbers. Count into the word or phrase by the
given number and record the letter in that position. For example: "2 Cake, 6 Pudding, 5
Shortening" gives you "ant".
- Alpha-numeric codes are also very common. If you end up with a list of numbers try replacing
the numbers with the corresponding letters like this: 1 = A, 2 = B, 3 = C... 26 = Z.
Occasionally, these types of codes will "wrap around", so don't despair if you see a
number greater than 26. Just subtract 26 and try again. In this scenario 27 (27-26 = 1) =
A, 28 (28-26 = 2) = B etc. If you try this and it doesn't work, try other numeric codes
such as ASCII.
- Often a puzzle repeats a strategy multiple times.

You will likely need to backtrack frequently, so make sure to write out your steps as you go.
If you get stuck, try to think of a new way to approach the puzzle. Try:
- Rereading the title and the flavor text. These are the most important hints about what type
of strategies, themes or cultural references might be used to solve the puzzle.
- Checking for pop culture references
- Checking for references to a song/poem/book/movie/TV show

For strings, examples of strategies you might try include:
- Alphabetizing
- Using leftover letters to spell something
- Rearranging the letters (aka anagrams or "transposing")
- Seeing if there are any acronyms
- Diagonalizing (taking the first letter of the first answer, the second letter of the second
answer, etc.)
- Looking for unusual letter frequencies
- Puns and homophones
- Shifting from letters to numbers

For numbers, try:
- Shifting from numbers to letters
- Using it as a phone number
- Treating numbers as dates
- Treating numbers as ASCII numbers
- Seeing if there are any strange sequences
- Seeing if prime numbers are involved

For images, try:
- Looking at it in a mirror
- Squinting at it from far away
- Tilting it
- Looking at it upside down
- Looking through it
- Transcribing it neatly
"""

MODEL_DIRECTORY = "/scratch/hengzhil/huggingface"

def get_model_local_path(model_name):
    model_paths = {
        "Qwen/QVQ-72B-Preview": "models--Qwen--QVQ-72B-Preview/",
        "Qwen/Qwen2.5-VL-72B-Instruct": "models--Qwen--Qwen2.5-VL-72B-Instruct/",
        "OpenGVLab/InternVL3-8B": "models--OpenGVLab--InternVL3-8B/",
        "OpenGVLab/InternVL3-78B": "models--OpenGVLab--InternVL3-78B/",
        "moonshotai/Kimi-VL-A3B-Thinking": "models--moonshotai--Kimi-VL-A3B-Thinking/",
    }
    if model_name in model_paths:
        model_cache_dir = Path(MODEL_DIRECTORY) / model_paths[model_name]
        with open(os.path.join(model_cache_dir, "refs", "main"), "r") as f:
            commit_hash = f.read().strip()
        snapshot_path = model_cache_dir / "snapshots" / commit_hash
        return snapshot_path
    else:
        return None

class EvalModel():
    temperature: float = 0.0
    max_image_size: int = 1024

    def resize_image(self, image: Image) -> Image:
        h, w = image.size
        if h <= self.max_image_size and w <= self.max_image_size:
            return image

        factor = self.max_image_size / max(h, w)
        h = round(h * factor)
        w = round(w * factor)
        print(dict(old=image.size, resized=(h, w)))
        if image.mode == "RGBA":
            image = image.convert("RGB")
        image = image.resize((h, w), Image.LANCZOS)
        return image

    def run(self, prompt: str, image: Image = None) -> str:
        raise NotImplementedError

    def setTemp(self, temp: float):
        if not (0.0 <= temp <= 2.0):
            raise ValueError("Temperature must be between 0.0 and 2.0")
        self.temperature = temp

    def getTemp(self) -> float:
        return self.temperature




class OpenAIModel(EvalModel):
    timeout: int = 60
    engine: str = ""
    temperature: float = 0.0
    client: Optional[OpenAI]
    system_prompt: str = PUZZLE_SYSTEM_PROMPT

    def load(self, system_prompt: str | None = None):    
        if self.client is None:
            load_dotenv()
            key = os.environ["MIT_OPENAI_API_KEY"]
            self.client = OpenAI(api_key=key, timeout=self.timeout)
            if system_prompt is not None:
                self.system_prompt = system_prompt

    def make_messages(self, prompt: str, image: Image = None) -> List[dict]:
        inputs = [{"type": "text", "text": prompt}]
        if image is not None:
            image_text = convert_image_to_text(self.resize_image(image))
            url = f"data:image/png;base64,{image_text}"
            inputs.append({"type": "image_url", "image_url": {"url": url}})

        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": inputs}
        ]

    def run(self, prompt: str, image: Image = None) -> str | None:
        self.load()

        output = ""
        error_message = "The response was filtered"

        while not output:
            try:
                response = self.client.chat.completions.create(
                    model=self.engine,
                    messages=self.make_messages(prompt, image),
                    temperature=self.temperature,
                    max_tokens=2048,
                    top_p=0.9
                )
                if response.choices[0].finish_reason == "content_filter":
                    raise ValueError(error_message)
                output = response.choices[0].message.content

            except Exception as e:
                print(e)
                if error_message in str(e):
                    output = error_message

        output = None
        
        messages = self.make_messages(prompt, image)

        try:
            response = self.client.chat.completions.create(
                model=self.engine,
                messages=messages,
                temperature=self.temperature,
            )
            print(response)
            output = response.choices[0].message.content


        except Exception as e:
            print(e)
            return None

        return output

    def run_few_shot(self, prompts: List[str], images: List[Image.Image]) -> str:
        self.load()
        output = ""
        error_message = "The response was filtered"
        content = []
        for i, p in enumerate(prompts):
            for value in self.make_messages(p, images[i])[0]["content"]:
                content.append(value)

        while not output:
            try:
                response = self.client.chat.completions.create(
                    model=self.engine,
                    messages=[{"role": "user", "content": content}],
                    temperature=self.temperature,
                    max_tokens=1024,
                    top_p=0.9
                )
                if response.choices[0].finish_reason == "content_filter":
                    raise ValueError(error_message)
                output = response.choices[0].message.content

            except Exception as e:
                print(e)
                if error_message in str(e):
                    output = error_message

            if not output:
                print("OpenAIModel request failed, retrying.")

        return output


class GPT4oModel(OpenAIModel):
    engine: str = "gpt-4o-2024-05-13"

    def load(self):
        if self.client is None:
            load_dotenv()
            key = os.environ["MIT_OPENAI_API_KEY"]
            self.client = OpenAI(api_key=key, timeout=self.timeout)



    
class Qwen2_5VLModel(EvalModel):
    model_path: str = "Qwen/Qwen2.5-VL-72B-Instruct"
    processor = None 
    # template: str = "USER: <image>\n{prompt}\nASSISTANT:"
    dtype: torch.dtype = torch.bfloat16
    model = None
    tokenizer  = None

    def load(self):
        if self.model is None:
            local_path = get_model_local_path(self.model_path)
            if local_path is None:
                local_path = self.model_path
            print(f"Loading model from {local_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                local_path,     
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                device_map="auto",
                trust_remote_code=True
            )
            print("Qwen2.5-VL-72B-Instruct model loaded successfully.")
            self.processor = AutoProcessor.from_pretrained(local_path)
            print("Qwen2.5-VL-72B-Instruct processor loaded successfully.")
            self.model.eval()

    def run(self, prompt: str, images: List) -> str:
        max_image_size: int = 1024
        def resize_image(image: Image) -> Image:
            h, w = image.size
            if h <= max_image_size and w <= max_image_size:
                return image

            factor = max_image_size / max(h, w)
            h = round(h * factor)
            w = round(w * factor)
            if image.mode == "RGBA":
                image = image.convert("RGB")
            image = image.resize((h, w), Image.LANCZOS)
            return image

        def make_messages():
            inputs = [{"type": "text", "text": prompt}]
            if images is not None:
                for image in images:
                    image_text = convert_image_to_text(resize_image(image))
                    url = f"data:image/png;base64,{image_text}"
                    inputs.append({"type": "image", "image": url})

            return [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": PUZZLE_SYSTEM_PROMPT}
                    ],
                },
                {
                    "role": "user", 
                    "content": inputs
                }
            ]
       
        messages = make_messages()

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        # Inference: Generation of the output
        generated_ids = self.model.generate(**inputs, max_new_tokens=4096)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        print(output_text)
        
        torch.cuda.empty_cache()
        
        return output_text


class QwenQvQModel(EvalModel):
    model_path: str = "Qwen/QVQ-72B-Preview"
    device: str = "cuda" 
    dtype: torch.dtype = torch.bfloat16
    model: Optional[AutoModelForCausalLM] = None
    tokenizer: Optional[AutoTokenizer] = None
    processor: Optional[AutoProcessor] = None 

    def load(self):
        if self.model is None:
            local_path = get_model_local_path(self.model_path)
            if local_path is None:
                local_path = self.model_path
                
            self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                local_path, 
                device_map="auto",
                torch_dtype=self.dtype,
                attn_implementation="flash_attention_2"
            )
            # default processer
            self.processor = AutoProcessor.from_pretrained("Qwen/QVQ-72B-Preview")
            self.model.eval()

    def run(self, prompt: str, images: List) -> str:
        max_image_size: int = 1024

        def resize_image(image: Image) -> Image:
            h, w = image.size
            if h <= max_image_size and w <= max_image_size:
                return image

            factor = max_image_size / max(h, w)
            h = round(h * factor)
            w = round(w * factor)
            if image.mode == "RGBA":
                image = image.convert("RGB")
            image = image.resize((h, w), Image.LANCZOS)
            return image
        
        def make_messages():
            inputs = [{"type": "text", "text": prompt}]
            if images is not None:
                for image in images:
                    image_text = convert_image_to_text(resize_image(image))
                    url = f"data:image/png;base64,{image_text}"
                    inputs.append({"type": "image", "image": url})

            return [
                {
                    "role": "system",
                    "content": [
                        {"type": "text", "text": PUZZLE_SYSTEM_PROMPT}
                    ],
                },
                {
                    "role": "user", 
                    "content": inputs
                }
            ]
       
        messages = make_messages()

        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, _ = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.device)

        # Inference: Generation of the output
        print("The model is generating the output...")
        
        with torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=8192)
        
        print("The model has generated the output, decoding...")
        
        decoded = self.processor.batch_decode(
            [ids[0][inputs.input_ids.shape[1]:]],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        del ids, inputs
        torch.cuda.empty_cache()
        
        return decoded


# Use for internvl3
def split_model(model_name):
    import math
    from transformers import AutoConfig
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

class InternVLModel(EvalModel):
    IMAGENET_MEAN = (0.485, 0.456, 0.406)
    IMAGENET_STD = (0.229, 0.224, 0.225)
    max_tokens=4096
    model_path = "OpenGVLab/InternVL3-78B"
    device: str = "cuda" 
    dtype: torch.dtype = torch.bfloat16
    model: Optional[AutoModelForCausalLM] = None
    tokenizer: Optional[AutoTokenizer] = None
    processor: Optional[AutoProcessor] = None 
    generation_config: dict = None

    def load(self):
         if self.model is None:
            local_path = get_model_local_path(self.model_path)
            if local_path is None:
                local_path = self.model_path
            device_map = split_model(local_path)
            self.model = AutoModel.from_pretrained(
                local_path,
                torch_dtype=self.dtype,
                low_cpu_mem_usage=True,
                device_map=device_map,
                trust_remote_code=True
            ).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(local_path, trust_remote_code=True)
            self.generation_config = dict(
                max_new_tokens=self.max_tokens,
                do_sample=(self.temperature != 0),
            )
            # default processer
            self.processor = AutoProcessor.from_pretrained(local_path, trust_remote_code=True)


    def run(self, prompt: str, images: List = None):
        if images is not None:
            all_pixel_values = []
            for image in images:
                # conversation = self._convert_conversation(conversation)
                pixel_values = self._load_image(image, max_num=6).to(torch.bfloat16).to(self.model.device)
                all_pixel_values.append(pixel_values)
                # num_patches_list.append(pixel_values.shape[0])
            pixel_values = torch.cat(all_pixel_values, dim=0)
            question = PUZZLE_SYSTEM_PROMPT + prompt + ("<image>" * len(images))
            response = self.model.chat(self.tokenizer, pixel_values, question, 
                                       self.generation_config)
        torch.cuda.empty_cache()
        return response
    
    def _build_transform(self, input_size):
        MEAN, STD = self.IMAGENET_MEAN, self.IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform


    def _find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio


    def _dynamic_preprocess(self, image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self._find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images


    def _load_image(self, image, input_size=448, max_num=6):
        image = image.convert('RGB')
        transform = self._build_transform(input_size=input_size)
        images = self._dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values
    
class KimiModel(EvalModel):
    model_path = "moonshotai/Kimi-VL-A3B-Thinking"
    template = "USER: <image>\n{prompt}\nASSISTANT:"
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16 
    model: Optional[AutoModelForCausalLM] = None
    tokenizer: Optional[AutoTokenizer] = None
    processor: Optional[AutoProcessor] = None

    def load(self):
        if self.model is None:
            local_path = get_model_local_path(self.model_path)
            if local_path is None:
                local_path = self.model_path
            print(f"Loading model from {local_path}")
            self.model = AutoModelForCausalLM.from_pretrained(
                local_path,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True,
            ).eval()
            self.processor = AutoProcessor.from_pretrained(
                local_path,
                trust_remote_code=True,
            )

    def run(self, prompt: str, images: List | None = None) -> str:
        max_image_size: int = 1024
        self.load()

        def resize_image(image: Image) -> Image:
            h, w = image.size
            if h <= max_image_size and w <= max_image_size:
                return image

            factor = max_image_size / max(h, w)
            h = round(h * factor)
            w = round(w * factor)
            if image.mode == "RGBA":
                image = image.convert("RGB")
            image = image.resize((h, w), Image.LANCZOS)
            return image

        def make_messages():
            inputs = []
            if images is not None:
                for image in images:
                    inputs.append({"type": "image", "image": image})
            inputs.append({"type": "text", "text": PUZZLE_SYSTEM_PROMPT + prompt})

            return [
                {
                    "role": "user",
                    "content": inputs
                }
            ]

        messages = make_messages()

        text = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
        images_resized = [resize_image(image) for image in images] if images is not None else None
        inputs = self.processor(
            images=images_resized,
            text=text,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=4096)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        print(output_text)
        torch.cuda.empty_cache()
        return output_text[0]



def select_model(model_name: str, **kwargs) -> EvalModel:
    model_map = dict(
        gpt4o=GPT4oModel,
        qwen_qvq=QwenQvQModel,
        qwen2_5=Qwen2_5VLModel,
        internvl3=InternVLModel,
        kimiv3=KimiModel,
    )
    model_class = model_map.get(model_name)
    if model_class is None:
        raise ValueError(f"{model_name}. Choose from {list(model_map.keys())}")
    return model_class()
