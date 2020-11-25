#!/bin/bash

python examples/conformer/train_conformer.py \
  --tfrecords \
  --devices 7 
  
#  --tbs 8 \
#  --ebs 8 \

  
