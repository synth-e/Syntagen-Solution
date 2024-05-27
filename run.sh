if [ ! -d ".tmp" ]; then
  mkdir .tmp
fi

gdown 1F00eHKJlE2jzDHMmBAee-AVQVBA1Bful -O .tmp/voc-classifier-weight.pth
python generate.py -c solution.yaml
rm -rf .tmp