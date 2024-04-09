#!/bin/bash

file=$1
precision=5

echo $file
echo "number of running words:"
cat $file | wc -w
echo "number of unique words:"
cat $file | tr -s ' ' '\n' | sort | uniq | wc -l
echo "average sentence length:"
echo "scale=$precision; $(cat $file | wc -w) / $(cat $file | wc -l)" | bc


