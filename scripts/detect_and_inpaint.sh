python3 -m src.detector --image_path $1 || python3 detector

bash lama/docker/2_predict.sh $(pwd)/lama/big-lama $(pwd)/test_images $(pwd)/output_images device=cpu