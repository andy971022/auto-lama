SHELL=/bin/bash
CONDA_ACTIVATE=source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate ; conda activate
CONDA_ENV_NAME=auto-lama

create-env:
	conda env export -n $(CONDA_ENV_NAME) > conda_env.yml

build-env:
	conda env remove --name $(CONDA_ENV_NAME)
	conda env create -f=conda_env.yml
	$(CONDA_ACTIVATE)
	bash scripts/build.sh

detect_and_inpaint:
	bash scripts/detect_and_inpaint.sh $(IMAGE_PATH)

clean:
	git clean -dfx