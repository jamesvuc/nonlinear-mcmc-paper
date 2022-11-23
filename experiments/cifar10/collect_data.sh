
for SEED in {1..5}; do
    # echo "SEED=$SEED"
    python3 cifar10_nonlin_resnet_mcmc.py --seed $SEED
done
