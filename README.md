# Simple Delf Sample

These delf files were originated from tensorflow research example,
but the code was shortened for the feasibility test.

original url
https://github.com/tensorflow/models/tree/master/research/delf/delf/python/detect_to_retrieve

## delf_gld_config.pbtxt

- scale : only works at 1. scale
- dimension : 30 dim
- regional search : disabled

## start.bat

Because similarity check takes more than 1500 hours, 
these batch file will execute delf_similarity2.py simultaneously.
If 23 jobs are executed simultaneously, time takes 80 hours (5% of original time spend)

## executing 
You can use 1) Jupyter Notebook, 2) py files

## result
Feasibility test executing the whole process of DELF whether to see it works on large dataset.
(110,000 test / 700,000 index approx.)

only 10% of test files was correctly submitted due to time limit, 

with restriction below
- scale down resolution to original file size of 25%, 
- search scale down to 10%, 
- disable regional search, 
- dimension 100 -> 30

records 0.0013 on 2019 Google Landmark Retrieval Kaggle Competition.
