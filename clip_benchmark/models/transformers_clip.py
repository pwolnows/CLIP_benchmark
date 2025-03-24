import torch
from torch import nn
from transformers import AutoModel, AutoProcessor
import onnxruntime as ort
import openvino as ov
from PIL import Image
import requests
from pathlib import Path


sample_path = Path("coco.jpg")
if not sample_path.exists():
    r = requests.get("https://storage.openvinotoolkit.org/repositories/openvino_notebooks/data/data/image/coco.jpg")
    with sample_path.open("wb") as f:
        f.write(r.content)

image = Image.open(sample_path)
input_labels = ["cat", "dog", "wolf", "tiger", "man", "horse", "frog", "tree", "house", "computer",]
text_descriptions = [f"This is a photo of a {label}" for label in input_labels]


class TransformerWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def encode_text(self, text):
        return self.model.get_text_features(**text)

    def encode_image(self, image):
        image = {key: value.squeeze(1) for key, value in image.items()}
        return self.model.get_image_features(**image)

def load_transformers_clip(model_name, pretrained, cache_dir, device):
    ckpt = f"{model_name}/{pretrained}"
    model = AutoModel.from_pretrained(ckpt, cache_dir=cache_dir, device_map=device)
    model = TransformerWrapper(model)

    processor = AutoProcessor.from_pretrained(ckpt)
    transforms = lambda image: processor(images=image, return_tensors="pt")
    tokenizer = lambda text: processor(text=text, padding="max_length", max_length=64, return_tensors="pt")
    return model, transforms, tokenizer


class OVWrapper:
    def __init__(self, model, inputs):
        self.model = model
        self.inputs = inputs

    def encode_text(self, text):
        input_dict = {k: v.cpu() for k, v in text.items()}
        if "pixel_values" not in input_dict:
            input_dict["pixel_values"] = self.inputs["pixel_values"]
        return torch.from_numpy(self.model(input_dict)[2]).to(self.model.device)

    def encode_image(self, image):
        # we get an extended dimension possibly due to the collation in dataloader
        input_dict = {k: v.squeeze(1).cpu() for k, v in image.items()}
        if "input_ids" not in input_dict:
            input_dict["input_ids"] = self.inputs["input_ids"]
        if "attention_mask" not in input_dict:
            input_dict["attention_mask"] = self.inputs["attention_mask"]
        return torch.from_numpy(self.model(input_dict)[3]).to(self.model.device)

    def eval(self):
        pass

def load_optimum_intel_clip(model_name, pretrained, cache_dir, device):
    ckpt = f"{model_name}/{pretrained}"
    processor = AutoProcessor.from_pretrained(ckpt)

    core = ov.Core()
    ov_path = f"{ckpt}/openvino_model.xml"
    compiled_model = core.compile_model(ov_path, "CPU")
    inputs = processor(text=text_descriptions, images=[image], return_tensors="pt", padding=True)
    model = OVWrapper(compiled_model, inputs)
    transforms = lambda image: processor(images=image, return_tensors="pt")
    tokenizer = lambda text: processor(text=text, padding="max_length", max_length=64, return_tensors="pt")
    return model, transforms, tokenizer


class ONNXWrapper:
    def __init__(self, model_path, inputs):
        self.session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.inputs = inputs

    def encode_text(self, text):
        input_dict = {k: v.cpu().numpy() for k, v in text.items()}
        if "pixel_values" not in input_dict:
            input_dict["pixel_values"] = self.inputs["pixel_values"].cpu().numpy()

        outputs = self.session.run(["onnx::MatMul_3309"], input_dict)
        return torch.from_numpy(outputs[0]).cuda()

    def encode_image(self, image):
        input_dict = {k: v.squeeze(1).cpu().numpy() for k, v in image.items()}
        if "input_ids" not in input_dict:
            input_dict["input_ids"] = self.inputs["input_ids"].cpu().numpy()
        if "attention_mask" not in input_dict:
            input_dict["attention_mask"] = self.inputs["attention_mask"].cpu().numpy()

        outputs = self.session.run(["onnx::Transpose_3303"], input_dict)
        return torch.from_numpy(outputs[0]).cuda()

    def eval(self):
        pass

def load_onnx_clip(model_name, pretrained, cache_dir, device):
    ckpt = f"{model_name}/{pretrained}"
    processor = AutoProcessor.from_pretrained(ckpt)

    onnx_path = f"{ckpt}/model.onnx"
    inputs = processor(text=text_descriptions, images=[image], return_tensors="pt", padding=True)
    model = ONNXWrapper(onnx_path, inputs)
    # core = ov.Core()
    # compiled_model = core.compile_model(onnx_path, "CPU")
    # model = OVWrapper(compiled_model, inputs)

    transforms = lambda image: processor(images=image, return_tensors="pt")
    tokenizer = lambda text: processor(text=text, padding="max_length", max_length=64, return_tensors="pt")
    return model, transforms, tokenizer
