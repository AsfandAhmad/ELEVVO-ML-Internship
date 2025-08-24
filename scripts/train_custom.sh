#!/usr/bin/env bash
python src/training/train.py --data_dir data/processed --model custom --img_size 64 --batch_size 64 --epochs 25 --learning_rate 0.001
