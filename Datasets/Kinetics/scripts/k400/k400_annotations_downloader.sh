#!/bin/bash

VID_DIR=$1
TARGZ_DIR=$2

# Download directories vars
root_dl="$VID_DIR"
root_dl_targz="$TARGZ_DIR"

# Make root directories
[ ! -d "$root_dl" ] && mkdir "$root_dl"
[ ! -d "$root_dl_targz" ] && mkdir "$root_dl_targz"

# Download annotations csv files
curr_dl=${root_dl}/annotations
url_tr=https://s3.amazonaws.com/kinetics/400/annotations/train.csv
url_v=https://s3.amazonaws.com/kinetics/400/annotations/val.csv
url_t=https://s3.amazonaws.com/kinetics/400/annotations/test.csv
[ ! -d "$curr_dl" ] && mkdir -p "$curr_dl"
wget -c $url_tr -P "$curr_dl"
wget -c $url_v -P "$curr_dl"
wget -c $url_t -P "$curr_dl"

# Download readme
url=http://s3.amazonaws.com/kinetics/400/readme.md
wget -c $url -P "$root_dl"

# Downloads complete
echo -e "\nDownloads complete!"
