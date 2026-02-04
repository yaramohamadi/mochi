import torch
from pipeline_stg_mochi import MochiSTGPipeline
from diffusers.utils import export_to_video
import os

os.makedirs("samples", exist_ok=True)

print("ok")

ckpt_path = "genmo/mochi-1-preview"
pipe = MochiSTGPipeline.from_pretrained(
    ckpt_path, variant="bf16", torch_dtype=torch.bfloat16
).to("cuda")

prompt = (
    "A close-up of a beautiful woman's face with colored powder exploding around her, creating an abstract splash of vibrant hues, realistic style."
)

Imaging Quality Aesthetic Quality Motion Smoothness Dynamic Degree Temporal Flickering

guidance_scale = 4.5
num_frames = 81
do_rescaling = False

# CFG only (disable STG)
stg_scale = 0.0
stg_applied_layers_idx = [34]  # irrelevant when stg_scale=0, but harmless

scenarios = [
    # {
    #     "name": "CFG_Dropout_short",
    #     "cfg_uncond_reference": "dropout-short",
    #     "cfg_anchor_interval": 25, 
    # },
    #     "name": "CFG",
    #     "cfg_uncond_reference": "standard",
    #     "cfg_anchor_interval": 1,  # ignored in "first"
    # },
    {
        "name": "CFG_sparse_guidance_short_anchor_every10",
        "cfg_uncond_reference": "nearest-short",
        "cfg_anchor_interval": 10,  # in OUTPUT frames
    }, 
    {
        "name": "CFG_sparse_guidance_short_anchor_every5",
        "cfg_uncond_reference": "nearest-short",
        "cfg_anchor_interval": 5,  # in OUTPUT frames
    }, 
    # {
    #     "name": "CFG_weighted_anchor_every10",
    #     "cfg_uncond_reference": "weighted",
    #     "cfg_anchor_interval": 10,  # in OUTPUT frames
    # },
    #     {
    #     "name": "CFG_weighted_anchor_every25",
    #     "cfg_uncond_reference": "weighted",
    #     "cfg_anchor_interval": 25,  # in OUTPUT frames
    # }, 
    #         {
    #     "name": "CFG_weighted_anchor_every25",
    #     "cfg_uncond_reference": "weighted",
    #     "cfg_anchor_interval": 25,  # in OUTPUT frames
    # }
]
print("ok")
for guided_ratio in [0.75]:
    for guidance_scale in [4.5]: # , 
        for sc in scenarios:
            generator = torch.Generator(device="cuda").manual_seed(42)
            print("ok?")
            frames = pipe(
                prompt,
                height=480,
                width=480,
                num_frames=num_frames,
                guidance_scale=guidance_scale,
                guided_ratio=guided_ratio,
                stg_applied_layers_idx=stg_applied_layers_idx,
                stg_scale=stg_scale,
                do_rescaling=do_rescaling,
                cfg_uncond_reference=sc["cfg_uncond_reference"],
                cfg_anchor_interval=sc["cfg_anchor_interval"],
                generator=generator,
            ).frames[0]

            cfg_anchor_interval = sc["cfg_anchor_interval"]

            video_name = (
                f"{sc['name']}_gs{guidance_scale}_every{cfg_anchor_interval}_gr{guided_ratio}_frames{num_frames}_rescale{do_rescaling}.mp4"
            )
            video_path = os.path.join("samples", video_name)
            export_to_video(frames, video_path, fps=30)
            print(f"Saved: {video_path}")
