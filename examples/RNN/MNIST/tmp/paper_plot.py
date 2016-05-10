"""
This file plots result of MNIST classification
"""


import einstein as e
from einstein.tools import load_pickle_file
import matplotlib.pyplot as plt

ndata = 100
nmiddle = 2
case = 3

gru_losses = load_pickle_file("gru_adam_type_255_titan_x_losses")[0:ndata:nmiddle]
gru_times = load_pickle_file("gru_adam_type_255_titan_x_times")[0:ndata:nmiddle]


sgu_losses = load_pickle_file("sgu_adam_type_255_more_iter_titan_x_losses")[0:ndata:nmiddle]
sgu_times = load_pickle_file("sgu_adam_type_255_more_iter_titan_x_times")[0:ndata:nmiddle]

dsgu_losses = load_pickle_file("dsgu_adam_type_255_more_iter_titan_x_losses")[0:ndata:nmiddle]
dsgu_times = load_pickle_file("dsgu_adam_type_255_more_iter_titan_x_times")[0:ndata:nmiddle]

lstm_losses = load_pickle_file("lstm_record_titan_x_losses")[0:ndata:nmiddle]
lstm_times = load_pickle_file("lstm_record_titan_x_times")[0:ndata:nmiddle]

irnn_losses = load_pickle_file("irnn_record_titan_x_losses")[0:ndata:nmiddle]
irnn_times = load_pickle_file("irnn_record_titan_x_times")[0:ndata:nmiddle]

print(sgu_times)

plt.figure(1)
plt.plot(xrange(len(dsgu_losses)), dsgu_losses,'->', label='DSGU', markevery=case)
plt.plot(xrange(len(irnn_losses)), irnn_losses, '-^', label='IRNN', markevery=case)
plt.plot(xrange(len(lstm_losses)), lstm_losses, '-o', label='LSTM', markevery=case)
plt.plot(xrange(len(sgu_losses)), sgu_losses, '-<', label= 'SGU', markevery=case)
plt.plot(xrange(len(gru_losses)), gru_losses, '-*', label='GRU', markevery=case)

plt.ylabel("Accuracy")
plt.xlabel("Iteration Number")
plt.legend(loc=4)
plt.savefig("mnist_example_iter.png")

plt.figure(2)
plt.plot(dsgu_times, dsgu_losses,'->', label='DSGU', markevery=case)
plt.plot(irnn_times, irnn_losses, '-^', label='IRNN', markevery=case)
plt.plot(sgu_times, sgu_losses, '-<', label= 'SGU', markevery=case)
plt.plot(lstm_times, lstm_losses, '-o', label='LSTM', markevery=case)
plt.plot(gru_times, gru_losses, '-*', label='GRU', markevery=case)

plt.ylabel("Accuracy")
plt.xlabel("Seconds")
plt.legend(loc=4)
plt.savefig("mnist_example_time.png")
