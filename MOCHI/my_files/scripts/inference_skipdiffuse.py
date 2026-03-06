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
   # {"name": "CFG_delta0_skip1", "cfg_uncond_delta_steps": 0, "skip_stride": 1},  # baseline
   # {"name": "CFG_delta1_skip1", "cfg_uncond_delta_steps": 1, "skip_stride": 1},
   # {"name": "CFG_delta2_skip1", "cfg_uncond_delta_steps": 2, "skip_stride": 1},
   # {"name": "CFG_delta4_skip1", "cfg_uncond_delta_steps": 4, "skip_stride": 1},
   # {"name": "CFG_delta5_skip1", "cfg_uncond_delta_steps": 5, "skip_stride": 1},
   # {"name": "CFG_delta6_skip1", "cfg_uncond_delta_steps": 6, "skip_stride": 1},
   # {"name": "CFG_delta7_skip1", "cfg_uncond_delta_steps": 7, "skip_stride": 1},
   # {"name": "CFG_delta8_skip1", "cfg_uncond_delta_steps": 8, "skip_stride": 1},
   # {"name": "CFG_delta9_skip1", "cfg_uncond_delta_steps": 9, "skip_stride": 1},
   # {"name": "CFG_delta10_skip1", "cfg_uncond_delta_steps": 10, "skip_stride": 1},

   {"name": "CFG_delta0_skip2", "cfg_uncond_delta_steps": 0, "skip_stride": 2},  
   {"name": "CFG_delta1_skip2", "cfg_uncond_delta_steps": 1, "skip_stride": 2},
   {"name": "CFG_delta2_skip2", "cfg_uncond_delta_steps": 2, "skip_stride": 2},
   {"name": "CFG_delta4_skip2", "cfg_uncond_delta_steps": 4, "skip_stride": 2},
   {"name": "CFG_delta5_skip2", "cfg_uncond_delta_steps": 5, "skip_stride": 2},
   {"name": "CFG_delta6_skip2", "cfg_uncond_delta_steps": 6, "skip_stride": 2},
   {"name": "CFG_delta7_skip2", "cfg_uncond_delta_steps": 7, "skip_stride": 2},
   {"name": "CFG_delta8_skip2", "cfg_uncond_delta_steps": 8, "skip_stride": 2},
   {"name": "CFG_delta9_skip2", "cfg_uncond_delta_steps": 9, "skip_stride": 2},
   {"name": "CFG_delta10_skip2", "cfg_uncond_delta_steps": 10, "skip_stride": 2},

   {"name": "CFG_delta0_skip3", "cfg_uncond_delta_steps": 0, "skip_stride": 3},  
   {"name": "CFG_delta1_skip3", "cfg_uncond_delta_steps": 1, "skip_stride": 3},
   {"name": "CFG_delta2_skip3", "cfg_uncond_delta_steps": 2, "skip_stride": 3},
   {"name": "CFG_delta4_skip3", "cfg_uncond_delta_steps": 4, "skip_stride": 3},
   {"name": "CFG_delta5_skip3", "cfg_uncond_delta_steps": 5, "skip_stride": 3},
   {"name": "CFG_delta6_skip3", "cfg_uncond_delta_steps": 6, "skip_stride": 3},
   {"name": "CFG_delta7_skip3", "cfg_uncond_delta_steps": 7, "skip_stride": 3},
   {"name": "CFG_delta8_skip3", "cfg_uncond_delta_steps": 8, "skip_stride": 3},
   {"name": "CFG_delta9_skip3", "cfg_uncond_delta_steps": 9, "skip_stride": 3},
   {"name": "CFG_delta10_skip3", "cfg_uncond_delta_steps": 10, "skip_stride": 3},

   # {"name": "CFG_delta0_skip3", "cfg_uncond_delta_steps": 0, "skip_stride": 3},
   # {"name": "CFG_delta0_skip4", "cfg_uncond_delta_steps": 0, "skip_stride": 4},
   # {"name": "CFG_delta0_skip5", "cfg_uncond_delta_steps": 0, "skip_stride": 5},
   # {"name": "CFG_delta0_skip6", "cfg_uncond_delta_steps": 0, "skip_stride": 6},
   # {"name": "CFG_delta0_skip7", "cfg_uncond_delta_steps": 0, "skip_stride": 7},
   # {"name": "CFG_delta0_skip8", "cfg_uncond_delta_steps": 0, "skip_stride": 8},
   # {"name": "CFG_delta0_skip9", "cfg_uncond_delta_steps": 0, "skip_stride": 9},
   # {"name": "CFG_delta0_skip10", "cfg_uncond_delta_steps": 0, "skip_stride": 10},

   # {"name": "CFG_delta2_skip3", "cfg_uncond_delta_steps": 2, "skip_stride": 3},
   # {"name": "CFG_delta3_skip4", "cfg_uncond_delta_steps": 3, "skip_stride": 4},
   # {"name": "CFG_delta4_skip5", "cfg_uncond_delta_steps": 4, "skip_stride": 5},
   # {"name": "CFG_delta5_skip6", "cfg_uncond_delta_steps": 5, "skip_stride": 6},
   # {"name": "CFG_delta6_skip7", "cfg_uncond_delta_steps": 6, "skip_stride": 7},
   # {"name": "CFG_delta7_skip8", "cfg_uncond_delta_steps": 7, "skip_stride": 8},
   # {"name": "CFG_delta8_skip9", "cfg_uncond_delta_steps": 8, "skip_stride": 9},
   # {"name": "CFG_delta9_skip10", "cfg_uncond_delta_steps": 9, "skip_stride": 10},
    

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
                cfg_uncond_delta_steps=sc["cfg_uncond_delta_steps"],
                skip_stride=sc["skip_stride"],
                cfg_uncond_delta_mode="pred_text",  # pred_text  # <-- REQUIRED
                cfg_uncond_delta_strength=1.0,       # optional, explicit
                            
                generator=generator,
            ).frames[0]

            video_name = (
                f"{sc['name']}_gs{guidance_scale}.mp4"
            )
            video_path = os.path.join("samples_pred_text_v_reverse/", video_name)
            export_to_video(frames, video_path, fps=30)
            print(f"Saved: {video_path}")