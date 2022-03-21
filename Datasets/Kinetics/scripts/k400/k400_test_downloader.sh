#!/bin/bash

VID_DIR=$1
TARGZ_DIR=$2

# Download directories vars
root_dl="$VID_DIR"
root_dl_targz="$TARGZ_DIR"

# Make root directories
[ ! -d "$root_dl" ] && mkdir "$root_dl"
[ ! -d "$root_dl_targz" ] && mkdir "$root_dl_targz"

# Download test tars, will resume
curr_dl=${root_dl_targz}/test
url=https://s3.amazonaws.com/kinetics/400/test/k400_test_path.txt
[ ! -d "$curr_dl" ] && mkdir -p "$curr_dl"
wget -c -i $url -P "$curr_dl"

# Downloads complete
echo -e "\nDownloads complete! Now run extractor, k400_test_extractor.sh"
