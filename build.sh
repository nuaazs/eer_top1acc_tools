#!/bin/bash
cd cpp
g++ calc_score.cpp -std=c++11 -mfma -O3 -Ofast -ffast-math -fopenmp -lopenblas -o calc
g++ get_top1_and_score.cpp -std=c++11 -mfma -O3 -Ofast -ffast-math -fopenmp -lopenblas -o top1
g++ get_top1_acc.cpp -std=c++17 -mfma -O3 -Ofast -ffast-math -fopenmp -lopenblas -o top1acc
g++ get_eer.cpp -std=c++11 -mfma -O3 -Ofast -ffast-math -fopenmp -lopenblas -o eer

cp calc ../utils
cp top1 ../utils
cp eer ../utils
cp top1acc ../utils
cd ../