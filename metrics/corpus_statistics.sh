#!/bin/bash

file=$1 # takes the first argument as the file name
precision=$2 # global parameter for precision of average sentence length

echo $file
echo "number of running words:"
cat $file | wc -w # use wc to count the words with -w option
echo "number of unique words:"
cat $file | tr -s ' ' '\n' | sort | uniq | wc -l # put every word into a line, sort them and remove duplicates with uniq -> then count the lines
#sed 's/ /\n/g' $file | sort | uniq -u | wc -w #replace spaces with \n (newline) sort the file to make it available for finding duplicates with uniq -u and then count the words
echo "average sentence length:"
echo "scale=$precision; $(cat $file | wc -w) / $(cat $file | wc -l)" | bc # calculate the number of words divided by the number of lines -> average sentence length
# the above line assumes that each line contains exactly one sentence (which is how it seems here)
