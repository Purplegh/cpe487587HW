#!/bin/bash


KEYWORD="hw02"

# Run training 5 times with same parameters
for i in {1..5}
do
    echo "Run $i / 5"
    python scripts/multiclass_impl.py \
        -f data/Android_Malware.csv \
        -e 10000\
        -l 0.001 \
        -k $KEYWORD
done

# Aggregate results and generate boxplot
python scripts/multiclass_eval.py -k $KEYWORD
