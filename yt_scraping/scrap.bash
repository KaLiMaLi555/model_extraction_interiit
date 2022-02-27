#!/bin/bash
NUM_VIDEOS=$1
LENGTH=$2
DATA_DIR=data
KINETICS400_DIR=$DATA_DIR/kinetics400
KINETICS600_DIR=$DATA_DIR/kinetics600

echo "Length of videos: $LENGTH"
echo "Number of videos: $NUM_VIDEOS"

echo "Creating directories..."
mkdir -p $KINETICS400_DIR
mkdir -p $KINETICS600_DIR

echo "Downloading Kinetics-400 videos"
cat kinetics_400_labels.csv|while read line
do
    while IFS=, read -r col1 col2
    do
        echo "Downloading videos for $col2"
        # Create a directory named col2
        mkdir -p $KINETICS400_DIR/$col2
        # Download the videos
        yt-dlp "ytsearch:$col2" --max-downloads $NUM_VIDEOS -P $KINETICS400_DIR/$col2 --match-filter "duration>15" --downloader ffmpeg --downloader-args "ffmpeg_i:-ss 00:00:00.00 -t 00:00:$LENGTH.00"
    done
done

echo "Downloading Kinetics-600 videos"
cat kinetics_600_labels.csv|while read line
do
    while IFS=, read -r col1 col2
    do
        echo "Downloading videos for $col2"
        # Create a directory named col2
        mkdir -p $KINETICS600_DIR/$col2
        # Download the videos
        yt-dlp "ytsearch:$col2" --max-downloads $NUM_VIDEOS -P $KINETICS600_DIR/$col2 --match-filter "duration>15" --downloader ffmpeg --downloader-args "ffmpeg_i:-ss 00:00:00.00 -t 00:00:$LENGTH.00"
    done
done

echo "Done!"