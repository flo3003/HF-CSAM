# This script computes the average training loss, training accuracy, validation loss and validation accuraccy over the 5 splits and saves the results in a new file

FILES=("cnn_cifar10" "mlp_cifar10" "mlp_mnist" "cnn_mnist")

for f in "${FILES[@]}" 

do

awk '{print $1, $2, $3, $4, $5}' $f"_1.csv" > sss

for i in `seq 6 11`

do

paste <(awk -v i="$i" '{print $i}' $f"_1.csv" ) <(awk -v i="$i" '{print $i}' $f"_3.csv" ) <(awk -v i="$i" '{print $i}' $f"_5.csv" ) <(awk -v i="$i" '{print $i}' $f"_7.csv" ) <(awk -v i="$i" '{print $i}' $f"_9.csv" ) | awk '{print ($1+$2+$3+$4+$5)/5}' > ttt

paste sss ttt | sed 's/\t/ /' > $f"_mean_values.csv"

cp $f"_mean_values.csv" sss

done

done

rm sss ttt

