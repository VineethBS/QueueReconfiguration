#!/usr/bin/python

import os
import numpy as np

seed = 1
max_buffer = 5000
num_queues = 2
max_iterations = 5000
debug = 0

######## Description of the policy ########
# policy = 1
# filename = "results_LCQ.csv"
# policy = 2
# filename = "results_EXH.csv"
# policy = 3
# filename = "results_LQ.csv"
# policy = 5
# filename = "results_EXP_Rule.csv"
policy = 6
filename = "results_VFMW_Rule.csv"

######## Description of the arrival process ########
distribution = 1

arrival_queue1_batchsize = 3
arrival_queue1_arrivalrate = 0.1
arrival_queue1_prob = arrival_queue1_arrivalrate / arrival_queue1_batchsize

arrival_queue2_batchsize = 3
arrival_queue2_arrivalrate = 0.2
arrival_queue2_prob = arrival_queue1_arrivalrate / arrival_queue1_batchsize

######## Description of the connection process ########
connection_queue1_prob = 0.5;
connection_queue2_prob = 0.5;

######## Parameter variation in experiment  ########
for arrival_queue2_arrivalrate in np.arange(0.2, 0.9, 0.1):
    arrival_queue2_prob = arrival_queue1_arrivalrate / arrival_queue1_batchsize
    command = "./simulation_iid %u %u %u %u %u %u %u %f %u %f %f %f %u %f %f >> %s"
    print command % (seed, max_buffer, num_queues, max_iterations, debug, policy, distribution, arrival_queue1_prob, arrival_queue1_batchsize, arrival_queue1_arrivalrate, connection_queue1_prob, arrival_queue2_prob, arrival_queue2_batchsize, arrival_queue2_arrivalrate, connection_queue2_prob, filename)
    os.system(command % (seed, max_buffer, num_queues, max_iterations, debug, policy, distribution, arrival_queue1_prob, arrival_queue1_batchsize, arrival_queue1_arrivalrate, connection_queue1_prob, arrival_queue2_prob, arrival_queue2_batchsize, arrival_queue2_arrivalrate, connection_queue2_prob, filename))


