git submodule update --init --recursive
export TORCH_HOME=$(pwd) && export PYTHONPATH=.
pip3 install wldhx.yadisk-direct
curl -L $(yadisk-direct https://disk.yandex.ru/d/ouP6l8VJ0HpMZg) -o big-lama.zip
unzip big-lama.zip -d lama/ || sudo unzip big-lama.zip -d lama/
