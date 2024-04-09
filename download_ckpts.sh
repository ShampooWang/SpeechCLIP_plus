# for downloading checkpoints
mkdir -p icassp_sasb_ckpts/SpeechCLIP+/base/flickr/cascaded+
wget https://huggingface.co/ShampooWang/speechclip_plus/resolve/main/base/flickr/cascaded%2B/epoch%3D326-step%3D38258-val_recall_mean_10%3D42.1100.ckpt?download=true -O icassp_sasb_ckpts/SpeechCLIP+/base/flickr/cascaded+/model.ckpt
mkdir -p icassp_sasb_ckpts/SpeechCLIP+/base/flickr/hybrid
wget https://huggingface.co/ShampooWang/speechclip_plus/resolve/main/base/flickr/hybrid/epoch%3D134-step%3D15794-val_recall_mean_10%3D80.0100.ckpt?download=true -O icassp_sasb_ckpts/SpeechCLIP+/base/flickr/hybrid/model.ckpt
mkdir -p icassp_sasb_ckpts/SpeechCLIP+/base/flickr/hybrid+
wget https://huggingface.co/ShampooWang/speechclip_plus/resolve/main/base/flickr/hybrid%2B/epoch%3D80-step%3D9476-val_recall_mean_10%3D81.0300.ckpt?download=true -O icassp_sasb_ckpts/SpeechCLIP+/base/flickr/hybrid+/model.ckpt

mkdir -p icassp_sasb_ckpts/SpeechCLIP+/large/flickr/cascaded
wget https://huggingface.co/ShampooWang/speechclip_plus/resolve/main/large/flickr/cascaded%2B/epoch%3D112-step%3D26441-val_recall_mean_10%3D60.0500.ckpt?download=true -O icassp_sasb_ckpts/SpeechCLIP+/large/flickr/cascaded/model.ckpt
mkdir -p icassp_sasb_ckpts/SpeechCLIP+/large/flickr/hybrid
wget https://huggingface.co/ShampooWang/speechclip_plus/resolve/main/large/flickr/hybrid/epoch%3D85-step%3D10061-val_recall_mean_10%3D90.1000.ckpt?download=true -O icassp_sasb_ckpts/SpeechCLIP+/large/flickr/hybrid/model.ckpt
mkdir -p icassp_sasb_ckpts/SpeechCLIP+/large/flickr/hybrid+
wget https://huggingface.co/ShampooWang/speechclip_plus/resolve/main/large/flickr/hybrid%2B/epoch%3D50-step%3D5966-val_recall_mean_10%3D89.3500.ckpt?download=true -O icassp_sasb_ckpts/SpeechCLIP+/large/flickr/hybrid+/model.ckpt

mkdir -p icassp_sasb_ckpts/SpeechCLIP+/large/coco/cascaded
wget https://huggingface.co/ShampooWang/speechclip_plus/resolve/main/large/coco/cascaded%2B/epoch%3D10-step%3D48740-val_recall_mean_10%3D31.2973.ckpt?download=true -O icassp_sasb_ckpts/SpeechCLIP+/large/coco/cascaded/model.ckpt
mkdir -p icassp_sasb_ckpts/SpeechCLIP+/large/coco/hybrid
wget https://huggingface.co/ShampooWang/speechclip_plus/resolve/main/large/coco/hybrid/epoch%3D12-step%3D28794-val_recall_mean_10%3D79.2988.ckpt?download=true -O icassp_sasb_ckpts/SpeechCLIP+/large/coco/hybrid/model.ckpt
mkdir -p icassp_sasb_ckpts/SpeechCLIP+/large/coco/hybrid+
wget https://huggingface.co/ShampooWang/speechclip_plus/resolve/main/large/coco/hybrid%2B/epoch%3D21-step%3D48729-val_recall_mean_10%3D83.5408.ckpt?download=true -O icassp_sasb_ckpts/SpeechCLIP+/large/coco/hybrid+/model.ckpt

echo "Done downloading all checkpoints"