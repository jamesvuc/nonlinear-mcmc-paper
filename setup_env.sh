
# install standard packages
pip install \
	numpy \
	matplotlib \
	tqdm

# install tensorflow datasets
# NOTE: you can install the regular tensorflow using `pip install tensorflow`
# if you want. We don't need GPU support for tf, so I install the cpu version
# which is python-dependent; see https://www.tensorflow.org/install/pip#package-location
pip install https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow_cpu-2.7.0-cp38-cp38-manylinux2010_x86_64.whl
pip install tensorflow-datasets

# install jax
pip install --upgrade \
	"jax[cuda]" \
	-f https://storage.googleapis.com/jax-releases/jax_releases.html

# # install jax-bayes
pip install git+https://github.com/jamesvuc/jax-bayes

# # install haiku
pip install dm-haiku

