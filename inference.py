# coding: utf-8
"""
for human
"""

import os
import glob
import os.path as osp
import tyro
import subprocess
import shutil
from src.config.argument_config import ArgumentConfig
from src.config.inference_config import InferenceConfig
from src.config.crop_config import CropConfig
from src.live_portrait_pipeline import LivePortraitPipeline
from src.utils.helper import basename


def partial_fields(target_class, kwargs):
    return target_class(**{k: v for k, v in kwargs.items() if hasattr(target_class, k)})


def fast_check_ffmpeg():
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except:
        return False


def fast_check_args(args: ArgumentConfig):
    if not osp.exists(args.source):
        raise FileNotFoundError(f"source info not found: {args.source}")
    if not osp.exists(args.driving):
        raise FileNotFoundError(f"driving info not found: {args.driving}")

def create_softlink(src, dst):
    if not osp.exists(dst):
        os.symlink(src, dst)


def main():
    # set tyro theme
    tyro.extras.set_accent_color("bright_cyan")
    args = tyro.cli(ArgumentConfig)

    ffmpeg_dir = os.path.join(os.getcwd(), "ffmpeg")
    if osp.exists(ffmpeg_dir):
        os.environ["PATH"] += (os.pathsep + ffmpeg_dir)

    if not fast_check_ffmpeg():
        raise ImportError(
            "FFmpeg is not installed. Please install FFmpeg (including ffmpeg and ffprobe) before running this script. https://ffmpeg.org/download.html"
        )

    fast_check_args(args)

    # specify configs for inference
    inference_cfg = partial_fields(InferenceConfig, args.__dict__)
    crop_cfg = partial_fields(CropConfig, args.__dict__)

    live_portrait_pipeline = LivePortraitPipeline(
        inference_cfg=inference_cfg,
        crop_cfg=crop_cfg
    )

    # run
    if args.flag_process_batch:
        objs = []
        fitting_obj_list_dir = args.batch_output_dir + 'fitting_obj_list_300.txt'
        print(f'{args.driving_option} {args.animation_region}')
        driving_imgs = glob.glob(os.path.join(args.batch_driving_dir, '*.png'))

        for root, dirs, files in os.walk(args.batch_source_dir):
            for dir_name in dirs:
                src_folder = osp.join(args.batch_source_dir, dir_name)
                source_imgs = glob.glob(os.path.join(src_folder, '*.png'))

                for driving_img in driving_imgs:
                    out_folder = osp.join(args.batch_output_dir, dir_name + '_' + basename(driving_img))
                    args.output_dir = out_folder
                    os.makedirs(out_folder, exist_ok=True)
                    objs.append(out_folder)

                    for source_img in source_imgs:
                        args.source = source_img
                        args.driving = driving_img
                        print(args.source, args.driving)

                        id = source_img.split('.')[0].split('_')[-1]
                        meta_name = f'metadata_{id}.json'
                        try:
                            live_portrait_pipeline.execute(args)
                        except Exception as e:
                            if str(e) == "No face detected in the source image!":
                                create_softlink(source_img, osp.join(out_folder, f'{basename(args.source)}.png'))
                            else:
                                raise
                        create_softlink(osp.join(src_folder, meta_name), osp.join(out_folder, meta_name))
        objs = sorted(objs, key=lambda x: (int(x.split('/')[-1].split('_')[0]), int(x.split('/')[-1].split('_')[1])))
        with open(fitting_obj_list_dir, 'w') as obj_list:
            for obj in objs:
                obj_list.write(obj + '\n')

    else:
        live_portrait_pipeline.execute(args)


if __name__ == "__main__":
    main()
