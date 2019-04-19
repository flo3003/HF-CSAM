export PYTHONHASHSEED=0

#-----------------------------------------------------------------
########################### PARAMETERS ###########################
#-----------------------------------------------------------------
OPTIMIZERS=("HFCSAM" "SGD" "Adam" "RMSprop")
DATA_AUGMENTATION=("f")

SGD_LEARNING_RATES=("0.01") 
HFCSAM_LEARNING_RATES=("0.05" "0.07" "0.1" "0.3" "0.5")
REST_LEARNING_RATES=("0.001") 

SGD_MOMENTUM_RATES=("0.7")
HFCSAM_XI_RATES=("0.8" "0.9" "0.99")

SEEDS=("1" "3" "5" "7" "9")

#-----------------------------------------------------------------#

for s in "${SEEDS[@]}"

do

for optimizer in "${OPTIMIZERS[@]}"

do

if [ "$optimizer" == "SGD" ]

then

LEARNING_RATES=(${SGD_LEARNING_RATES[@]})
MOMENTUM_RATES=(${SGD_MOMENTUM_RATES[@]})

elif [ "$optimizer" == "HFCSAM" ]

then

LEARNING_RATES=(${HFCSAM_LEARNING_RATES[@]})
MOMENTUM_RATES=(${HFCSAM_XI_RATES[@]})

else

LEARNING_RATES=(${REST_LEARNING_RATES[@]})
MOMENTUM_RATES=("0")

fi

for lr in "${LEARNING_RATES[@]}"

do

for m in "${MOMENTUM_RATES[@]}"

do

for aug in "${DATA_AUGMENTATION[@]}"

do

echo cnn, cifar, $optimizer, $lr, $m, $aug, $s
python cnn/cifar10_cnn_test_optimizers.py $optimizer $lr $m $aug $s > temp.txt

tail -n 1 temp.txt >> batch_results/cnn_cifar10_$s.csv
tail -n 1 batch_results/cnn_cifar10_$s.csv

echo mlp, cifar, $optimizer, $lr, $m, $aug, $s
python mlp/cifar10_mlp_test_optimizers.py $optimizer $lr $m $aug $s > temp.txt

tail -n 1 temp.txt >> batch_results/mlp_cifar10_$s.csv
tail -n 1 batch_results/mlp_cifar10_$s.csv

done

echo cnn, mnist, $optimizer, $lr, $m, $s
python cnn/mnist_cnn_test_optimizers.py $optimizer $lr $m $s > temp.txt

tail -n 1 temp.txt >> batch_results/cnn_mnist_$s.csv
tail -n 1 batch_results/cnn_mnist_$s.csv

echo mlp, mnist, $optimizer, $lr, $m, $s
python mlp/mnist_mlp_test_optimizers.py $optimizer $lr $m $s> temp.txt

tail -n 1 temp.txt >> batch_results/mlp_mnist_$s.csv
tail -n 1 batch_results/mlp_mnist_$s.csv

done

done

done

done
