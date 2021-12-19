python3 detector.py --image_path $1 || python3 detector.py

bash lama/docker/2_predict.sh $(pwd)/lama/big-lama $(pwd)/test_images $(pwd)/output device=cpu