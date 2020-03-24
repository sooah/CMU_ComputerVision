#!/bin/bash

if [ ! -f data.zip ]; then
    wget http://www.cs.cmu.edu/afs/cs/user/lkeselma/www/16720a_data/data.zip
fi
if [ ! -f images.zip ]; then
    wget http://www.cs.cmu.edu/afs/cs/user/lkeselma/www/16720a_data/images.zip
fi
unzip data.zip -d ../data/
unzip images.zip -d ../images/
rm images.zip
rm data.zip
