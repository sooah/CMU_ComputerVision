#!/bin/bash

wget http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz
wget http://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat
wget http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat
tar -xf 102flowers.tgz
python flowers_partition.py
rm 102flowers.tgz
rm setid.mat
rm imagelabels.mat
rm -rf jpg
