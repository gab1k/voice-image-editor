from pathlib import Path
from typing import Optional, Union

import torch
from PIL import Image
from diffusers import (
    StableDiffusionInstructPix2PixPipeline,
    StableDiffusionImg2ImgPipeline,
    AutoPipelineForImage2Image,
)
from datasets import load_dataset


class DiffusionImageEditor:
    SUPPORTED_TYPES = ("instruct_pix2pix", "sd_img2img", "sdxl_turbo")

    def __init__(
        self,
        model_type: str = "instruct_pix2pix",
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        num_inference_steps: int = 20,
        strength: float = 0.75,
        image_guidance_scale: float = 1.5,
        guidance_scale: float = 7.5,
        max_side: Optional[int] = 1024,
    ) -> None:
        if model_type not in self.SUPPORTED_TYPES:
            raise ValueError(f"Неизвестный model_type='{model_type}'")
        self.device = device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.model_type = model_type
        if model_name is None:
            if model_type == "instruct_pix2pix":
                model_name = "timbrooks/instruct-pix2pix"
            elif model_type == "sd_img2img":
                model_name = "runwayml/stable-diffusion-v1-5"
            elif model_type == "sdxl_turbo":
                model_name = "stabilityai/sdxl-turbo"
        self.model_name = model_name

        self.num_inference_steps = num_inference_steps
        self.strength = strength
        self.image_guidance_scale = image_guidance_scale
        self.guidance_scale = guidance_scale
        self.max_side = max_side

        print("Инициализация DiffusionImageEditor")
        self._load_pipeline()

    def _load_pipeline(self) -> None:
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32

        if self.model_type == "instruct_pix2pix":
            pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                safety_checker=None,
            )
        elif self.model_type == "sd_img2img":
            pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                safety_checker=None,
            )
        elif self.model_type == "sdxl_turbo":
            # SDXL-turbo через AutoPipelineForImage2Image
            pipe = AutoPipelineForImage2Image.from_pretrained(
                self.model_name,
                torch_dtype=torch_dtype,
                variant="fp16" if self.device == "cuda" else None,
            )

        self.pipeline = pipe.to(self.device)

        if self.device == "cuda":
            self.pipeline.enable_attention_slicing()

        print(f"Модель загружена: {self.model_name}")

    def edit(
        self,
        image: Union[str, Path, Image.Image],
        instruction: str,
        **override_params,
    ) -> Image.Image:

        if isinstance(image, (str, Path)):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise TypeError(
                f"image должен быть str, Path или PIL.Image.Image, получен {type(image)}"
            )

        # Даунскейлим очень большие изображения, чтобы не ловить CUDA OOM
        if self.max_side is not None:
            w, h = image.size
            max_dim = max(w, h)
            if max_dim > self.max_side:
                scale = self.max_side / float(max_dim)
                new_w = int(round(w * scale))
                new_h = int(round(h * scale))
                image = image.resize((new_w, new_h), Image.BICUBIC)

        steps = override_params.get("num_inference_steps", self.num_inference_steps)
        strength = override_params.get("strength", self.strength)
        img_guidance = override_params.get(
            "image_guidance_scale", self.image_guidance_scale
        )
        txt_guidance = override_params.get("guidance_scale", self.guidance_scale)

        if self.model_type == "instruct_pix2pix":
            out = self.pipeline(
                prompt=instruction,
                image=image,
                num_inference_steps=steps,
                image_guidance_scale=img_guidance,
                guidance_scale=txt_guidance,
            ).images[0]
        elif self.model_type == "sd_img2img":
            out = self.pipeline(
                prompt=instruction,
                image=image,
                num_inference_steps=steps,
                strength=strength,
                guidance_scale=txt_guidance,
            ).images[0]
        elif self.model_type == "sdxl_turbo":
            out = self.pipeline(
                prompt=instruction,
                image=image,
                num_inference_steps=steps,
                strength=strength,
                guidance_scale=txt_guidance,
            ).images[0]

        return out

    def close(self) -> None:
        if hasattr(self, "pipeline") and self.pipeline is not None:
            del self.pipeline
        if self.device == "cuda":
            torch.cuda.empty_cache()



if __name__ == "__main__":
    ds = load_dataset("arood0/mmm_project_with_audio_ru_final", split="train")
    editor = DiffusionImageEditor(
        model_type="instruct_pix2pix",
        model_name="timbrooks/instruct-pix2pix",
        device="cuda",
        num_inference_steps=20,
        strength=0.75,
        image_guidance_scale=1.5,
        guidance_scale=7.5,
        max_side=1024,
    )
    img = editor.edit(image=ds[0]["INPUT_IMG"], instruction=ds[0]["EDITING_INSTRUCTION_RU"])
    img.show()
    img.save(f"{ds[0]['IMAGE_ID']}.png")
