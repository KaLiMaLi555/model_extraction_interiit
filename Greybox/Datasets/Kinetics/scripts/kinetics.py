import os
from tqdm import tqdm
import argparse
import subprocess
import pathlib as Path
import shutil
from concurrent.futures import ProcessPoolExecutor


def call_bash(dir_path: str,
              set_name: str = "k400",
              typ: str = "downloader",
              part: str = "full",
              max_workers: int = 8):

    vid_path = os.path.join(dir_path, set_name)
    targz_path = os.path.join(vid_path, f"{set_name}_targz")
    errors = 0
    this_dir_name, _ = os.path.split(os.path.abspath(__file__))
    file_name = f"{set_name}_{part}_{typ}.sh"
    script_path = os.path.join(this_dir_name, set_name, file_name)
    script_path = script_path + " %s %s "
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            executor.map(subprocess.check_call(script_path % (vid_path, targz_path), shell=True))
    except subprocess.CalledProcessError:
        errors += 1
    if errors > 0:
        print(f'{errors} failed!!!!')


def extrapolate(input_dir, output_dir, out_frames: int = 16):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    error_count = 0
    videos = sorted(os.listdir(input_dir))

    for video in tqdm(videos):

        frames = sorted(os.listdir(input_dir / video))
        if len(frames) == 0:
            print(f'----> Skipping {video}: video has no frames')
            continue
        else:
            frames = sorted(frames * (out_frames // len(frames)))
            new_vid_frames = frames
            length = len(new_vid_frames)
            add_frames = out_frames % length
            x = length // (add_frames + 1)
            a = x

        if add_frames % 2 != 0:
            new_vid_frames.append(frames[length // 2])
            add_frames = add_frames - 1

        while add_frames != 0:
            new_vid_frames.append(frames[a - 1])
            new_vid_frames.append(frames[length - a])
            a = a + x
            add_frames = add_frames - 2

        new_vid_frames.sort()

        out_path = output_dir / video
        out_path.mkdir(parents=True, exist_ok=True)

        for idx, frame in enumerate(new_vid_frames):
            src = input_dir / video / frame
            dst = output_dir / video / (str(idx) + ".jpg")
            shutil.copy(src, dst)
        if len(os.listdir(output_dir / video)) != 16:
            print(len(new_vid_frames))
            error_count += 1
    if error_count > 0:
        print(f'----> {error_count} videos were not copied')
    else:
        print('----> All videos were copied')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Download kinetics dataset')
    parser.add_argument('--dir_path', type=str, default='Datasets/Kinetics')
    parser.add_argument('--set_name', type=str, default='k400')
    parser.add_argument('--part', type=str, default='full')
    parser.add_argument('--max_workers', type=int, default=8)
    parser.add_argument('--extract_frames', type=bool, default=True)
    parser.add_argument('--extend, type=int, default=0')
    args = parser.parse_args()

    call_bash(args.dir_path, args.set_name, "downloader", args.part, args.max_workers)
    call_bash(args.dir_path, args.set_name, "extractor", args.part, args.max_workers)

    if args.extract_frames:
        input_dir_path = os.path.join(args.dir_path, args.set_name, "test")
        input_dir_list = os.listdir(input_dir_path)
        output_dir_path = os.path.join(args.dir_path, f"{args.set_name}_{args.part}_frames_extracted")
        os.mkdir(output_dir_path)

        this_dir_name, _ = os.path.split(os.path.abspath(__file__))
        file_name = "extract_frames.sh"
        script_path = os.path.join(this_dir_name, args.set_name, file_name)
        script_path = script_path + " %s %s %s"
        errors = 0
        for vid in tqdm(input_dir_list):
            try:
                with ProcessPoolExecutor() as executor:
                    executor.map(subprocess.check_call(script_path % (vid, input_dir_path, output_dir_path),
                                                       shell=True))
            except subprocess.CalledProcessError:
                errors += 1
        if errors > 0:
            print(f'Failed to extract {errors} videos!!!!')

        if args.extend > 0:
            output_dir_path = os.path.join(args.dir_path, f"{args.set_name}_{args.part}_{args.extend}_frames_extended")
            os.mkdir(output_dir_path)
            extrapolate(os.path.join(args.dir_path, f"{args.set_name}_{args.part}_frames_extracted"), output_dir_path,
                        args.extend)