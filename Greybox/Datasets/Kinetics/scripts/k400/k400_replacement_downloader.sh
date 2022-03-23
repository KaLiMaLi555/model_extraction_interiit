#!/bin/bash

VID_DIR=$1
TARGZ_DIR=$2

# Download directories vars
root_dl="$VID_DIR"
root_dl_targz="$TARGZ_DIR"

# Make root directories
[ ! -d "$root_dl" ] && mkdir "$root_dl"
[ ! -d "$root_dl_targz" ] && mkdir "$root_dl_targz"

# Download replacement tars, will resume
curr_dl=${root_dl_targz}/replacement
url=https://s3.amazonaws.com/kinetics/400/replacement_for_corrupted_k400.tgz
[ ! -d "$curr_dl" ] && mkdir -p "$curr_dl"
wget -c $url -P "$curr_dl"

# Downloads complete
echo -e "\nDownloads complete! Now run extractor, k400_replacement_extractor.sh"