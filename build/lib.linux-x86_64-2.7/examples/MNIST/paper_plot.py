"""
This file plots result of MNIST classification
"""


import einstein as e
from einstein.tools import load_pickle_file
import matplotlib.pyplot as plt

sgu_losses = load_pickle_file("sgu_adam_type_255_more_iter_titan_x_losses")
sgu_times = load_pickle_file("sgu_adam_type_255_more_iter_titan_x_times")

dsgu_losses = load_pickle_file("dsgu_adam_type_255_more_iter_titan_x_losses")
dsgu_times = load_pickle_file("dsgu_adam_type_255_more_iter_titan_x_times")

lstm_losses = load_pickle_file("lstm_record_titan_x_losses")
lstm_times = load_pickle_file("lstm_record_titan_x_times")

irnn_losses = load_pickle_file("irnn_record_titan_x_losses")
irnn_times = load_pickle_file("irnn_record_titan_x_times")

print(sgu_times)

plt.figure(1)
plt.plot(xrange(len(dsgu_losses)), dsgu_losses,'r',
         xrange(len(irnn_losses)), irnn_losses, 'g',
         xrange(len(sgu_losses)), sgu_losses, 'orange',
         xrange(len(lstm_losses)), lstm_losses, 'k')
plt.ylabel("Accuracy")
plt.xlabel("Iteration Number")

plt.savefig("mnist_example_iter.png")

plt.figure(2)
plt.plot(dsgu_times, dsgu_losses,'r',
         irnn_times, irnn_losses, 'g',
         sgu_times, sgu_losses, 'orange',
         lstm_times, lstm_losses, 'k')
plt.ylabel("Accuracy")
plt.xlabel("Seconds")
plt.savefig("mnist_example_time.png")