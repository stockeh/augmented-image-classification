#!/bin/bash

# Perform augmented training and then email

if [ "$#" -ne 1 ]
then
  echo "First argument should be the email you want to be notified at"
  exit 1
fi

python3 ./train_augmented.py

echo "Your job is done." | mail -s "(`hostname`) Augmented Training Job DONE" -a ./training-out.csv $1
