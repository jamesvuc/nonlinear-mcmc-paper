
# install pytorch
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 \
	--extra-index-url https://download.pytorch.org/whl/cu113

# install jax
pip install --upgrade "jax[cuda]" \
	-f https://storage.googleapis.com/jax-releases/jax_releases.html

# install jax-bayes
pip install git+https://github.com/jamesvuc/jax-bayes

# install other libraries
pip install numpy matplotlib tqdm dm-haiku dm-tree