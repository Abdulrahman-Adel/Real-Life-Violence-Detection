#!/bin/bash

rm -r experiments
#--evaluate : t / true
#--submit : t / true
python -W ignore train.py -cd configs/test_configs -e t > log.txt

grep -Eo "Test.*" log.txt