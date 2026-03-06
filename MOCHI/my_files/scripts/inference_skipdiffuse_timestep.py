import torch
from pipeline_stg_mochi_skipdiffuse import MochiSTGPipeline
from diffusers.utils import export_to_video
import os

os.makedirs("samples", exist_ok=True)

ckpt_path = "genmo/mochi-1-preview"
pipe = MochiSTGPipeline.from_pretrained(
    ckpt_path, variant="bf16", torch_dtype=torch.bfloat16
).to("cuda")

prompt = (
    "A close-up of a beautiful woman's face with colored powder exploding around her, creating an abstract splash of vibrant hues, realistic style."
)

guidance_scale = 7.5# 4.5
num_frames = 81
do_rescaling = False

# CFG only (disable STG)
stg_scale = 0.0
stg_applied_layers_idx = [34]  # harmless when stg_scale=0

# ✅ NEW: test different δ (in *inference-step index* units)
scenarios = [
   {"name": "CFG_delta0_skip2_timestepGS1", "cfg_uncond_delta_steps": 0, "skip_stride": 2, "timestepGS": 1},  
   {"name": "CFG_delta1_skip2_timestepGS1", "cfg_uncond_delta_steps": 1, "skip_stride": 2, "timestepGS": 1},
   {"name": "CFG_delta2_skip2_timestepGS1", "cfg_uncond_delta_steps": 2, "skip_stride": 2, "timestepGS": 1},
   {"name": "CFG_delta4_skip2_timestepGS1", "cfg_uncond_delta_steps": 4, "skip_stride": 2, "timestepGS": 1},

   {"name": "CFG_delta1_skip2_timestepGS0.5", "cfg_uncond_delta_steps": 1, "skip_stride": 2, "timestepGS": 0.5},
   {"name": "CFG_delta2_skip2_timestepGS0.5", "cfg_uncond_delta_steps": 2, "skip_stride": 2, "timestepGS": 0.5},
   {"name": "CFG_delta4_skip2_timestepGS0.5", "cfg_uncond_delta_steps": 4, "skip_stride": 2, "timestepGS": 0.5},

   {"name": "CFG_delta1_skip2_timestepGS0.2", "cfg_uncond_delta_steps": 1, "skip_stride": 2, "timestepGS": 0.2},
   {"name": "CFG_delta2_skip2_timestepGS0.2", "cfg_uncond_delta_steps": 2, "skip_stride": 2, "timestepGS": 0.2},
   {"name": "CFG_delta4_skip2_timestepGS0.2", "cfg_uncond_delta_steps": 4, "skip_stride": 2, "timestepGS": 0.2},
]

for guided_ratio in [0.75]:
    for guidance_scale in [4.5]:
        for sc in scenarios:
            generator = torch.Generator(device="cuda").manual_seed(42)

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

                # keep these "standard" so ONLY your delta change is active
                cfg_uncond_reference="standard",
                cfg_anchor_interval=1,
                
                # ✅ THIS is your change (must exist in pipeline __call__)
                cfg_uncond_delta_steps=0, # sc["cfg_uncond_delta_steps"],
                skip_stride=sc["skip_stride"],
                cfg_uncond_delta_mode="pred_text",  # pred_text  # <-- REQUIRED
                cfg_uncond_delta_strength=1.0,       # optional, explicit
                timestep_guidance_scale=sc["timestepGS"],          # turn it ON (try 0.05–0.2)
                timestep_guidance_norm=True,
                timestep_guidance_delta_steps=sc["cfg_uncond_delta_steps"],
                generator=generator,
            ).frames[0]

            video_name = f"skip{sc['skip_stride']}_d{sc['cfg_uncond_delta_steps']}_tgs{sc['timestepGS']}_gs{guidance_scale}.mp4"
            video_path = os.path.join("samples_pred_text_v_timestepguidance_norm/", video_name)
            export_to_video(frames, video_path, fps=30)
            print(f"Saved: {video_path}")