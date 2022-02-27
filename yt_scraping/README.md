# Youtube Scraping

Scrap youtube videos of fixed length with the keywords of Kinetics-400 and Kinetics-600 labels present in `kinetics_400_labels.csv` and `kinetics_600_labels.csv` files.

## Requirements

```bash
pip install yt-dlp
```

## Usage

```bash
bash scrap.bash NUM_VIDEOS LENGTH
```

The script will download `NUM_VIDEOS` number of videos of length `LENGTH` seconds for each of the class labels present in the Kinetics-400 and Kinetics-600 in the corresponding directories named after labels.

`data` directory after downloading the videos:

```
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