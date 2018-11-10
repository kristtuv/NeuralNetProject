"""Needs description"""

import sys
sys.append('network')
sys.append('../')
sys.append('../../')

import numpy as np
from logreg import LogReg_Ising
import time

num_batches = [1, 100, 500, 1000, 65000]
rates = [0.001, 0.01, 0.1]
n = len(num_batches)
r = len(rates)

sizes = np.zeros((n,r))
train_accs = np.zeros((n,r))
test_accs = np.zeros((n,r))
train_costs = np.zeros((n,r))
test_costs = np.zeros((n,r))
times = np.zeros((n,r))

model = LogReg_Ising()
J = model.J         # Save the initialization

for i in range(n):
    for j in range(r):

        model.J = J      # Use the same initialization

        t_start = time.clock()

        model.optimize(m = num_batches[i], epochs = 80, eta = rates[j])

        t_stop = time.clock()
        times[i,j] = (t_stop - t_start)

        sizes[i,j] = model.batchSize
        train_accs[i,j] = model.train_accuracy
        test_accs[i,j] = model.test_accuracy
        train_costs[i,j] = model.train_cost
        test_costs[i,j] = model.test_cost

        print("\n\n")


print("Batch size|  L.rate | Train Cost| Test Cost | Train Acc | Test Acc |  Wall Time")
print("-"*80)
for i in range(n):
    for j in range(r):
        print(" %7g  | %7g |  %7g  |  %7g  |  %7g | %7g | %7g" %(sizes[i,j], \
                rates[j], train_costs[i,j], test_costs[i,j], train_accs[i,j], \
                test_accs[i,j], times[i,j]))
    print("-"*80)
