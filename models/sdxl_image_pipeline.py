from pathlib import Path

import torch
from diffusers import StableDiffusionXLPipeline
from ip_adapter.ip_adapter_faceid import IPAdapterFaceIDXL

from ..exceptions.binary_not_found import BinaryNotFound

from helpers.package import find_package_name

class SDXLImagePipeline():
    """
        Image pipeline used for IP Generation
    """

    def __init__(
                self,
                model_dir,
                model_name,
                vae_model=None,
                ip_checkpoint="ip-adapter-faceid_sdxl.bin",
                noise_scheduler_ddim=None,
                diffuser=StableDiffusionXLPipeline
                ):
        self._get_system_architecture()
        ip_checkpoint_path = Path(f"{find_package_name('IP-Adapter-Plugin').path}/lib/ip-adapter-faceid/{ip_checkpoint}")
        if not ip_checkpoint_path.exists:
            raise BinaryNotFound(f"Could not find path {ip_checkpoint_path}.\nPlease ensure you installed `lib/ip-adapter-faceid/`")
        self.model_name = model_name
        if model_dir[-1] != '/':
            model_dir += '/'
        self.model_dir = model_dir
        self.noise_scheduler_ddim = noise_scheduler_ddim
        self.vae_model = vae_model
        self.pipeline = self._generate_pipeline(diffuser)
        
        self.ip_model = IPAdapterFaceIDXL(self.pipeline, ip_checkpoint_path, self.to_value)
    
    def _get_system_architecture(self):
        # Default to CPU
        input_touch_type=torch.float32
        input_variant="fp32"
        to_value = "cpu"

        if torch.cuda.is_available():
            input_touch_type=torch.float16
            input_variant="fp16"
            to_value = "cuda"
        
        elif torch.backends.mps.is_available():
            input_touch_type=torch.float16
            input_variant="fp16"
            to_value = "mps"

        self.input_touch_type = input_touch_type
        self.input_variant = input_variant
        self.to_value = to_value

    def _generate_pipeline(self, diffuser):
        pipeline = diffuser.from_pretrained(
                self.model_dir+self.model_name,
                torch_dtype=self.input_touch_type, 
                variant=self.input_variant,
                use_safetensors=True,
                load_safety_checker=False,
                local_files_only=True
                )
        return pipeline

    def create_image(
                    self,
                    prompt,
                    width,
                    height,
                    face_embedding,
                    seed=2023,
                    guidance_scale=10,
                    num_inference_steps=50):

        return self.ip_model.generate(
            num_samples = 1,
            prompt = prompt,
            width = width,
            faceid_embeds=face_embedding,
            height = height,
            num_inference_steps = num_inference_steps,
            seed = seed,
            guidance_scale=guidance_scale
        )
