#!/bin/bash
MODE=$1
DATA_DIR=data

echo "MODE: $MODE"

if [ $MODE = "kinetics400" ]; then
    ROOT_DIR=$DATA_DIR/kinetics400
    VIDEO_DIR=$ROOT_DIR/video
    TRIMMED_VIDEO_DIR=$ROOT_DIR/video_trimmed
    FRAMES_DIR=$ROOT_DIR/frames
    echo "Creating directories..."
    mkdir -p $VIDEO_DIR
    mkdir -p $TRIMMED_VIDEO_DIR
    mkdir -p $FRAMES_DIR
    cat kinetics400_test_new.csv|while read line
    do
        while IFS=, read -r col1 col2 col3 col4 col5
        do
            echo "Downloading videos for $col2"
            yt-dlp "\"$col2\"" --output $VIDEO_DIR/$col2.mp4 --default-search "ytsearch" -S ext:mp4:m4a
            echo "Trimming videos for $col2"
            ffmpeg -i $VIDEO_DIR/$col2.mp4 -ss $col3 -to $col4 $TRIMMED_VIDEO_DIR/$col2.mp4
            echo "Extracting frames for $col2"
            SUB_FRAMES_DIR=$FRAMES_DIR/$col2
            mkdir -p $SUB_FRAMES_DIR
            ffmpeg -i "$TRIMMED_VIDEO_DIR/$col2.mp4" -r 30 -q:v 1 "$SUB_FRAMES_DIR/%06d.jpg"
        done
    done
elif [ $MODE = "kinetics600" ]; then
    ROOT_DIR=$DATA_DIR/kinetics600
    VIDEO_DIR=$ROOT_DIR/video
    TRIMMED_VIDEO_DIR=$ROOT_DIR/video_trimmed
    FRAMES_DIR=$ROOT_DIR/frames
    echo "Creating directories..."
    mkdir -p $VIDEO_DIR
    mkdir -p $TRIMMED_VIDEO_DIR
    mkdir -p $FRAMES_DIR
    cat kinetics600_test_new.csv|while read line
    do
        while IFS=, read -r col1 col2 col3 col4 col5
        do
            echo "Downloading videos for $col2"
            yt-dlp "\"$col2\"" --output $VIDEO_DIR/$col2.mp4 --default-search "ytsearch" -S ext:mp4:m4a
            echo "Trimming videos for $col2"
            ffmpeg $VIDEO_DIR/$col2.mp4 -ss $col3 -to $col4 $TRIMMED_VIDEO_DIR/$col2.mp4 
            echo "Extracting frames for $col2"
            SUB_FRAMES_DIR=$FRAMES_DIR/$col2
            mkdir -p $SUB_FRAMES_DIR
            ffmpeg -i "$TRIMMED_VIDEO_DIR/$col2.mp4" -r 30 -q:v 1 "$SUB_FRAMES_DIR/%06d.jpg"
        done
    done
else
    echo "Invalid mode"
fi

echo "Done!"