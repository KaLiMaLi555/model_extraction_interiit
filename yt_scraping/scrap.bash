#!/bin/bash
MODE=$1
NUM_VIDEOS=$2
MIN_DURATION=$3
MAX_DURATION=$4
DATA_DIR=data
KINETICS400_DIR=$DATA_DIR/kinetics400
KINETICS600_DIR=$DATA_DIR/kinetics600

echo "MODE: $MODE"
echo "Length of videos: $LENGTH"

if [ $MODE = "kinetics400" ]; then
    echo "Creating directories..."
    mkdir -p $KINETICS400_DIR
    echo "Downloading Kinetics-400 videos"
    cat kinetics_400_labels.csv|while read line
    do
        while IFS=, read -r col1 col2
        do
            echo "Downloading videos for $col2"
            # Create a directory named col2
            mkdir -p $KINETICS400_DIR/$col2
            # Download the videos
            yt-dlp --max-downloads $NUM_VIDEOS -P $KINETICS400_DIR/$col2 --match-filter "duration>$MIN_DURATION & duration<$MAX_DURATION" "ytsearchall:$col2" 
        done
    done
elif [ $MODE = "kinetics600" ]; then
    echo "Creating directories..."
    mkdir -p $KINETICS600_DIR
    echo "Downloading Kinetics-600 videos"
    cat kinetics_600_labels.csv|while read line
    do
        while IFS=, read -r col1 col2
        do
            echo "Downloading videos for $col2"
            # Create a directory named col2
            mkdir -p $KINETICS600_DIR/$col2
            # Download the videos
            yt-dlp --max-downloads $NUM_VIDEOS -P $KINETICS600_DIR/$col2 --match-filter "duration>$MIN_DURATION & duration<$MAX_DURATION" "ytsearchall:$col2" 
        done
    done
else
    echo "Invalid mode"
fi

echo "Done!"