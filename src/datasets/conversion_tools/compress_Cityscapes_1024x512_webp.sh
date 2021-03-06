# Compress the Cityscapes dataset from original 2048x1024 PNG to 1024x512 WEBP
# $DIR_CITYSCAPES = location of original dataset
# $DIR_CITYSCAPES_SMALL = output location

DIR_CITYSCAPES=/home/waywaybao_cs10/leftImg8bit_trainvaltest
DIR_CITYSCAPES_SMALL=/home/waywaybao_cs10/leftImg8bit_trainvaltest_small

python compress_images.py \
	$DIR_CITYSCAPES/images/leftImg8bit \
	$DIR_CITYSCAPES_SMALL/images/leftImg8bit \
	"cwebp {src} -o {dest} -q 90 -sharp_yuv -m 6 -resize 1024 512" \
	--ext ".webp" --concurrent 20

python compress_images.py \
	$DIR_CITYSCAPES/gtFine \
	$DIR_CITYSCAPES_SMALL/gtFine \
	"convert {src} -filter point -resize 50% {dest}" \
	--ext ".png" --concurrent 20
