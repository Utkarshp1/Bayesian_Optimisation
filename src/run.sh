#!/bin/bash
python3 main.py -e topk -ed ./ -obj_fn $1 -m min -d 4 -zacq EI -o 1 -q topk -gacq SumGradientAcquisitionFunction -b 100 -r 2 -nm 0.0 -nv 0.01 -nr 10 -rs 32 -s 42 -i 5