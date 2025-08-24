#!/usr/bin/env bash
python src/training/train.py --data_dir data/processed --model mobilenet --img_size 96 --batch_size 64 --epochs 10 --learning_rate 0.0005 --freeze_base True
