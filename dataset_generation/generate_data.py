import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--examples', type=int, help='No. of videos to be generated.')
    parser.add_argument('--frames', type=int, default='5', help='No. of frames per video')
    parser.add_argument('--h', type=int, default='224', help='height of frame in px')
    parser.add_argument('--w', type=int, default='224', help='width of frame in px')
    opts = parser.parse_args()

    # Parameters
    n_videos, n_frames = opts.examples, opts.frames
    shape = (opts.h, opts.w, 3)

    # Output directory setup
    out_path = Path.cwd() / 'data'
    out_path.mkdir(parents=True, exist_ok=True)

    # For each 'video' we need to generate
    for video_id in range(n_videos):
        # Create the video output directory
        img_folder_path = out_path / str(video_id)
        img_folder_path.mkdir(parents=True, exist_ok=True)

        # For each frame in the video
        for frame_id in range(n_frames):
            # Generate random noise using numpy, rescale, and convert to uint8
            frame = 255 * np.random.rand(*shape)
            frame = np.rint(frame).astype('uint8')

            # Save as a png using PIL
            Image.fromarray(frame).save(img_folder_path / f'{frame_id}.png')


if __name__ == '__main__':
    main()
