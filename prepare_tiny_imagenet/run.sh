#!/usr/bin/env bash
TARGET_DIR=../tiny-imagenet-200
if [ ! -d "$TARGET_DIR" ]; then
	echo "Downloading Tiny-ImageNet..."
	wget -nc http://cs231n.stanford.edu/tiny-imagenet-200.zip
	unzip tiny-imagenet-200.zip
fi

echo "Preparing Tiny-ImageNet..."
rm -r ./tiny-imagenet-200/test
python val_format.py

find . -name "*.txt" -delete

python resize.py

mv ./tiny-imagenet-200 "$TARGET_DIR"

# create soft link for hat code
HAT_TARGET_DIR="../hat/dat/tiny-imagenet-200"
if [ ! -d "$HAT_TARGET_DIR" ]; then
	mkdir -p "../hat/dat"
	ln -s "$(realpath $TARGET_DIR)" "$(realpath $HAT_TARGET_DIR)"
fi

