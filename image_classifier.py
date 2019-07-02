#MNIST dataset
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


from sklearn.datasets import fetch_mldata
dataset = fetch_mldata('MNIST original')

X = dataset.data
y = dataset.target

some_digit = X[12]