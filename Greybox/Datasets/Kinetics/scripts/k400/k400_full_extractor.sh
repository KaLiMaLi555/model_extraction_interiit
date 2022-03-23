#!/bin/bash

VID_DIR=$1
TARGZ_DIR=$2

# Download directories vars
root_dl="$VID_DIR"
root_dl_targz="$TARGZ_DIR"

# Make root directories
[ ! -d "$root_dl" ] && mkdir "$root_dl"

# Extract train
curr_dl=$root_dl_targz/train
curr_extract=$root_dl/train
# shellcheck disable=SC2086
[ ! -d "$curr_extract" ] && mkdir -p $curr_extract
tar_list=$(ls "$curr_dl")
for f in $tar_list
do
	[[ $f == *.tar.gz ]] && echo Extracting "$curr_dl"/"$f" to "$curr_extract" && tar zxf "$curr_dl"/"$f" -C "$curr_extract"
done

# Extract validation
curr_dl=$root_dl_targz/val
curr_extract=$root_dl/val
[ ! -d "$curr_extract" ] && mkdir -p "$curr_extract"
tar_list=$(ls "$curr_dl")
for f in $tar_list
do
	[[ $f == *.tar.gz ]] && echo Extracting "$curr_dl"/"$f" to "$curr_extract" && tar zxf "$curr_dl"/"$f" -C "$curr_extract"
done

# Extract test
curr_dl=$root_dl_targz/test
curr_extract=$root_dl/test
[ ! -d "$curr_extract" ] && mkdir -p "$curr_extract"
tar_list=$(ls "$curr_dl")
for f in $tar_list
do
	[[ $f == *.tar.gz ]] && echo Extracting "$curr_dl"/"$f" to "$curr_extract" && tar zxf "$curr_dl"/"$f" -C "$curr_extract"
done

# Extract replacement
curr_dl=$root_dl_targz/replacement
curr_extract=$root_dl/replacement
[ ! -d "$curr_extract" ] && mkdir -p "$curr_extract"
tar_list=$(ls "$curr_dl")
for f in $tar_list
do
	[[ $f == *.tgz ]] && echo Extracting "$curr_dl"/"$f" to "$curr_extract" && tar zxf "$curr_dl"/"$f" -C "$curr_extract"
done

# Extraction complete
echo -e "\nExtractions complete!"