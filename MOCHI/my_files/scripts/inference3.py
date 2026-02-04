import torch
from pipeline_stg_mochi import MochiSTGPipeline
from diffusers.utils import export_to_video
import os

os.makedirs("samples", exist_ok=True)

prompts_dir = "VBench/prompts/prompts_per_dimension"
output_base_dir = "/home/ymbahram/scratch/videos_mochi/vbench_evaluation"
# dimension_list = ["imaging_quality", "aesthetic_quality", "motion_smoothness", "dynamic_degree", "temporal_flickering"]
dimension_list = ["overall_consistency", "subject_consistency", "temporal_flickering"]
# overall_consistency: aesthetic_quality, imaging_quality		93
# subject_consistency: motion_smoothness, dynamic_degree		72
# temporal_flickering: temporal_flickering			75


 ckpt_path = "genmo/mochi-1-preview"
pipe = MochiSTGPipeline.from_pretrained(
    ckpt_path, variant="bf16", torch_dtype=torch.bfloat16
).to("cuda")

guidance_scale = 4.5
num_frames = 81
do_rescaling = False

# CFG only (disable STG)
stg_scale = 0.0
stg_applied_layers_idx = [34]  # irrelevant when stg_scale=0, but harmless

for dimension in dimension_list:

    torch.manual_seed(42)

    # read prompt list
    #with open(f'VBench/prompts/prompts_per_dimension/{dimension}.txt', 'r') as f:
    #    prompt_list = f.readlines()
    #prompt_list = [prompt.strip() for prompt in prompt_list]

    prompt_list = [
    "A close-up of a beautiful woman's face with colored powder exploding around her, creating an abstract splash of vibrant hues, realistic style."
    ]



    for prompt in prompt_list:

        scenarios = [
            {
                "cfg_uncond_reference": "standard",
                "cfg_anchor_interval": 0,  # in OUTPUT frames
            }, 
            # {
            #     "cfg_uncond_reference": "nearest-short",
            #     "cfg_anchor_interval": 10,  # in OUTPUT frames
            # }, 
            #{
            #    "cfg_uncond_reference": "nearest-short",
            #    "cfg_anchor_interval": 5,  # in OUTPUT frames
            #}, 
        ]

        for guided_ratio in [0.75]:
            for guidance_scale in [4.5]: # , 
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
                        cfg_uncond_reference=sc["cfg_uncond_reference"],
                        cfg_anchor_interval=sc["cfg_anchor_interval"],
                        generator=generator,
                        num_videos_per_prompt=5,
                    ).frames

                    cfg_anchor_interval = sc["cfg_anchor_interval"]
                    cfg_uncond_reference = sc["cfg_uncond_reference"]

                    output_dir_name = (
                        f"{output_base_dir}/{dimension}/gs{guidance_scale}_mode{cfg_uncond_reference}_every{cfg_anchor_interval}_frames{num_frames}"
                    )
                    for index in range(len(frames)):
                        os.makedirs(output_dir_name, exist_ok=True)
                        video_name = f"{output_dir_name}/{prompt}-{index}.mp4"
                        video_path = os.path.join(output_dir_name, video_name)
                        export_to_video(frames[index], video_path, fps=30)
                        print(f"Saved: {video_path}")
