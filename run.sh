#!/bin/bash
if [ $1 = 1 ]
then
  python ./Q1/naive_bayes.py $2 $3 $4
elif [ $1 = 2 ]
then
  python ./Q2/lwlr.py $2 $3 $4 $5
fi


#python Q$1.py $2 $3 $4 $5 $6
