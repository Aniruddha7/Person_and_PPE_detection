import splitfolders
import os
path = "D:\PPE_detection\datasets"
print(os.listdir(path))
splitfolders.ratio(path,seed=1337, output="train_valid_test", ratio=(0.7, 0.2, 0.1))
