if [ ! -d ".tmp" ]; then
  mkdir .tmp
fi

if [ ! -f ".tmp/voc-classifier-weight.pth" ]; then
  gdown 1F00eHKJlE2jzDHMmBAee-AVQVBA1Bful -O .tmp/voc-classifier-weight.pth
fi

if [ ! -f ".tmp/vitb16.pt" ]; then
  curl https://openaipublic.azureedge.net/clip/models/5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f/ViT-B-16.pt -o .tmp/vitb16.pt
fi

python generate.py -c solution.yaml