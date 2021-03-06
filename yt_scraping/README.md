# Youtube Scraping

Scrap youtube videos of fixed length with the keywords of Kinetics-400 and Kinetics-600 labels present in `kinetics_400_labels.csv` and `kinetics_600_labels.csv` files.

## Requirements

```bash
pip install yt-dlp
```

## Usage

```bash
bash scrap.bash MODE NUM_VIDEOS MIN_DURATION MAX_DURATION
```

Specify the `MODE` of downloading the videos as `kinetics400` or `kinetics600` for Kinetics-400 and 600 respectively. The script will download `NUM_VIDEOS` number of videos with constrains of `MIN_DURATION` and `MAX_DURATION` for each of the class labels present in the Kinetics-400 and Kinetics-600 in the corresponding directories named after labels.

`data` directory after downloading the videos:

```bash
data
└───kinetics400
    └───abseiling
            video_1
            video_2
            ...
    └───acting in play
            video_1
            video_2
            ...
 
└───kinetics600
    └───abseiling
            video_1
            video_2
            ...
    └───acting in play
            video_1
            video_2
            ...
```

After downloading the raw videos, trim the videos into 10 seconds clips from the 100th to 110th seconds of the original video clip using the following command:

```bash
bash trim.bash MODE DATA_DIR START_TIME LENGTH
```

Provide the appropriate `MODE` (`kinetics400` or `kinetics600`), `DATA_DIR` where the raw videos are present, `START_TIME` from which the videos are to be trimmed and `LENGTH` of which the videos must be trimmed.