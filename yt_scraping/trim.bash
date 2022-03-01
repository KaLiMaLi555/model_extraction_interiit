#!/bin/bash
MODE=$1
DATA_DIR=$2
DATA_DIR_10=${DATA_DIR}_10
KINETICS400_DIR=$DATA_DIR/kinetics400
KINETICS600_DIR=$DATA_DIR/kinetics600
KINETICS400_DIR_10=$DATA_DIR_10/kinetics400
KINETICS600_DIR_10=$DATA_DIR_10/kinetics600

echo "MODE: $MODE"
echo "Creating directories..."
mkdir -p $DATA_DIR_100
mkdir -p $KINETICS400_DIR_10
mkdir -p $KINETICS600_DIR_10

if [ $MODE = "kinetics400" ]; then
    cat kinetics_400_labels.csv|while read line
    do
        while IFS=, read -r col1 col2
        do
            echo "Trimming videos for $col2"
            OUT_DATA_DIR=${KINETICS400_DIR_10}/${col2}
            mkdir -p $OUT_DATA_DIR
            for video in $(ls -A1 -U ${KINETICS400_DIR}/${col2}/*)
                do
                    out_name="${OUT_DATA_DIR}/${video##*/}"
                    if [ ! -f "${out_name}" ]; then
                        ffmpeg -ss 100 -t 10 -i "${video}" "${out_name}"
                    fi
                done
        done
    done
elif [ $MODE = "kinetics600" ]; then
    cat kinetics_600_labels.csv|while read line
    do
        while IFS=, read -r col1 col2
        do
            echo "Trimming videos for $col2"
            OUT_DATA_DIR=${KINETICS600_DIR_10}/${col2}
            mkdir -p $OUT_DATA_DIR
            for video in $(ls -A1 -U ${KINETICS600_DIR}/${col2}/*)
                do
                    out_name="${OUT_DATA_DIR}/${video##*/}"
                    if [ ! -f "${out_name}" ]; then
                        ffmpeg -ss 100 -t 10 -i "${video}" "${out_name}"
                    fi
                done
        done
    done
else
    echo "Invalid mode"
fi

echo "Done!"