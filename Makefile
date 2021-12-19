CONDA_ENV_NAME=auto-lama

create-conda-env:
	conda env export -n $(CONDA_ENV_NAME) > conda_env.yml
	pip list --format=freeze > requirements.txt

build-conda-env:
	conda env remove --name $(CONDA_ENV_NAME)
	conda env create -f=conda_env.yml

build-env:
	bash scripts/build.sh

detect_and_inpaint: 
	bash scripts/detect_and_inpaint.sh $(IMAGE_PATH)

clean:
	git clean -dfx

