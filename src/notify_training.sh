#!/bin/bash

# Perform augmented training and then email

if [ "$#" -lt 2 ]
then
  echo "First argument should be the email you want to be notified at. Second argument should be the script you want to run"
  exit 1
fi

python3 $2
# ./train_augmented.py

if [ -n "$3" ]
then
  echo "Your job is done." | mail -s "(`hostname`) Job DONE" -a $3 $1
else
  echo "Your job is done." | mail -s "(`hostname`) Job DONE" $1
fi
