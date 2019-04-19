#!/bin/bash


> train.list
> test.list
COUNT=-1
FID=-1
for folder in $1/frames/*
do
    FID=$[$FID + 1]
    for imagesFolder in "$folder"/*
    do
        COUNT=$[$COUNT + 1]
        if (($COUNT % $2 > 0))
        then
            echo "$imagesFolder" $COUNT $FID >> train.list
        else
            echo "$imagesFolder" $COUNT $FID >> test.list
        fi
    done
done
