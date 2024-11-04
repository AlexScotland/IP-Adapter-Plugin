import io

from fastapi import APIRouter, Response, UploadFile, Depends, File
from diffusers import DDIMScheduler, AutoencoderKL, StableDiffusionXLPipeline
from PIL import Image

from settings import MODEL_DIRECTORY

from ..models.serializers.base_image import BaseImage
from ..factories.face_embedding_factory import FaceEmbeddingFactory
from ..models.sdxl_image_pipeline import SDXLImagePipeline


from routers.v2 import __clean_up_pipeline


ROUTER = APIRouter(
    prefix="/ip_adapter",
    tags=["Dedicated IP Adapters Plugin for same character image generation"]
)

@ROUTER.post("/generate/")
def generate(image_payload: BaseImage = Depends(), uploaded_image: UploadFile=File(...)):
    uploaded_file_data = io.BytesIO(uploaded_image.file.read())
    facial_embedding = FaceEmbeddingFactory.create(image_payload.face_analysis_model, uploaded_file_data)

    noise_scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        clip_sample=False,
        set_alpha_to_one=False,
        steps_offset=1,
    )

    image_pipeline = SDXLImagePipeline(
        MODEL_DIRECTORY,
        image_payload.model,
        noise_scheduler_ddim=noise_scheduler,
        diffuser=StableDiffusionXLPipeline
    )

    image_store = io.BytesIO()
    image = image_pipeline.create_image(
        prompt=image_payload.prompt,
        height=image_payload.height,
        width=image_payload.width,
        face_embedding=facial_embedding)
    image[0].save(image_store,"png")
    # Cleanup our pipeline
    __clean_up_pipeline(image_pipeline)

    return Response(content=image_store.getvalue(), media_type="image/png")