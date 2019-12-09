import numpy as np
import math
import random
from utils import *

# the arrival time of the next request
def nextTime(arrival_rate):
	return -math.log(1.0 - random.random()) / arrival_rate

def genRequests(num_requests, arrival_rate):
    new_req_seq = []
    curr_time = 0.0
    frame_time = 0.0   

    for i in range(num_requests):
    	if curr_time < frame_time:
    		curr_time = frame_time
    	temp = nextTime(arrival_rate)
    	curr_time = curr_time + temp
    	new_req_seq.append(curr_time)
    	frame_time += float(random.randint(1,10))/FRAME_RATE
    return new_req_seq

# generating random user requests for single model scheduling
def genNewReq():
	arrival_rate = 1.0/0.02
	all_requests = []
	for i in range(NUM_USERS):
		curr_requests = genRequests(500, arrival_rate)
		all_requests.extend(curr_requests)
	all_requests.sort()

	return all_requests[0:NUM_NEW_REQUEST]
