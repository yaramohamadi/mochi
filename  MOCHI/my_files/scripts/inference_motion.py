import os
import csv
import torch
from diffusers.utils import export_to_video
from pipeline_stg_mochi import MochiSTGPipeline

os.makedirs("samples", exist_ok=True)

ckpt_path = "genmo/mochi-1-preview"
pipe = MochiSTGPipeline.from_pretrained(
    ckpt_path, variant="bf16", torch_dtype=torch.bfloat16
).to("cuda")

prompt = (
    "A close-up of a beautiful woman's face with colored powder exploding around her, "
    "creating an abstract splash of vibrant hues, realistic style."
)

num_frames = 81
height = 480
width = 480
do_rescaling = False

# disable STG for now 
stg_scale = 0.0
stg_applied_layers_idx = [34]

scenarios = [
    {"name": "CFG_Dropout", "cfg_uncond_reference": "standard", "cfg_anchor_interval": 25},
]

guided_ratios = [1.0]
guidance_scales = [4.5, 7.5]

# Motion guidance sweep
motion_scales = [0.75, 1, 1.5, 2]
motion_detach_prev = True
motion_only_from_frame = 1

# logging
csv_path = os.path.join("samples", "sweep_log.csv")
write_header = not os.path.exists(csv_path)

with open(csv_path, "a", newline="") as f:
    writer = csv.writer(f)
    if write_header:
        writer.writerow([
            "video_path",
            "scenario",
            "guidance_scale",
            "guided_ratio",
            "motion_scale",
            "seed",
            "cfg_uncond_reference",
            "cfg_anchor_interval",
            "num_frames",
            "height",
            "width",
            "do_rescaling",
        ])

    seed = 42
    for sc in scenarios:
        for gr in guided_ratios:
            for gs in guidance_scales:
                for ms in motion_scales:

                    # deterministic per run
                    generator = torch.Generator(device="cuda").manual_seed(seed)

                    # filename (make it easy to parse)
                    video_name = (
                        f"{sc['name']}"
                        f"_gs{gs:.2f}"
                        f"_gr{gr:.2f}"
                        f"_ms{ms:.2f}"
                        f"_frames{num_frames}"
                        f"_rescale{int(do_rescaling)}"
                        f"_seed{seed}.mp4"
                    )
                    video_path = os.path.join("samples", video_name)

                    # skip if already rendered
                    if os.path.exists(video_path):
                        print(f"[skip] {video_path}")
                        continue

                    print(f"[run] sc={sc['name']} gs={gs} gr={gr} ms={ms}")

                    out = pipe(
                        prompt,
                        height=height,
                        width=width,
                        num_frames=num_frames,
                        guidance_scale=gs,
                        guided_ratio=gr,
                        stg_applied_layers_idx=stg_applied_layers_idx,
                        stg_scale=stg_scale,
                        do_rescaling=do_rescaling,
                        cfg_uncond_reference=sc["cfg_uncond_reference"],
                        cfg_anchor_interval=sc["cfg_anchor_interval"],
                        motion_scale=ms,
                        motion_detach_prev=motion_detach_prev,
                        motion_only_from_frame=motion_only_from_frame,
                        generator=generator,
                    )

                    frames = out.frames[0]
                    export_to_video(frames, video_path, fps=30)
                    print(f"Saved: {video_path}")

                    writer.writerow([
                        video_path,
                        sc["name"],
                        gs,
                        gr,
                        ms,
                        seed,
                        sc["cfg_uncond_reference"],
                        sc["cfg_anchor_interval"],
                        num_frames,
                        height,
                        width,
                        do_rescaling,
                    ])
                    f.flush()