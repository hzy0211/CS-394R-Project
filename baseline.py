import numpy as np
from utils import *
from System_Status import System_Status
from Read_Layer import Read_Layer

def read_layer(curr_status, data_file):
    #print("batch size: ",BATCH_SIZE)
    #print("num_shared_layers: ",curr_status.num_shared_layers)
    #print("group_num_t: ",group_num_t, "group_num_shared: ",GROUP_NUM_SHARED)
    curr_sum = 0.0
    fp = open(data_file,"r")
    for i in range(BATCH_SIZE):
        curr_layer = fp.readline().strip()
        curr_layer = curr_layer.split()
        for j in range(LAYER_SIZE):
            curr_status.batch_matrix[i][j] = round(float(curr_layer[j]),6)
            if i==0:
                curr_sum += curr_status.batch_matrix[i][j]

    fp.close()
    group_num = GROUP_NUM
    j = 0
    for k in range(BATCH_SIZE):
        curr_status.group_batch_matrix[k][j] = 0.0

    for i in range(LAYER_SIZE):
        for k in range(BATCH_SIZE):
            curr_status.group_batch_matrix[k][j] += curr_status.batch_matrix[k][i]

        if (curr_status.group_batch_matrix[0][j]>=curr_sum/(1.0*group_num)):
            curr_sum -= curr_status.group_batch_matrix[0][j]
            group_num -= 1
            if abs(curr_sum)<1e-8:
                curr_sum=0.0
            #print("curr_sum: ",curr_sum,"group_num: ",group_num,"i: ",i, "num_shared_layers: ", curr_status.num_shared_layers)
            assert (group_num>=0)
            assert (curr_sum>=0)
            j += 1

def compute_time(start_idx, end_idx, batch_size, curr_status):
	time = 0
	for i in range(start_idx, end_idx, 1):
		time += curr_status.group_batch_matrix[batch_size-1][i]
	return time

def always_batching(curr_status, new_req_seq):
	time_stamp = 0.0
	wait_time = 0.0
	wait_job = 0
	i = 0
	while i < NUM_NEW_REQUEST or np.array(curr_status.group_batch).sum() != 0:
		if (wait_job > 0 and np.array(curr_status.group_batch).sum() < MEMORY_SIZE) or (i < NUM_NEW_REQUEST and new_req_seq[i] <= time_stamp):
			if np.array(curr_status.group_batch).sum() >= MEMORY_SIZE:
				wait_job += 1
				i += 1
			elif wait_job > 0:
				curr_status.group_batch[0] += 1
				wait_job -= 1
			else:
				curr_status.group_batch[0] += 1
				i += 1
		elif np.array(curr_status.group_batch).sum() == 0:
			time_stamp = max(time_stamp, new_req_seq[i])
		else:
			for j in range(GROUP_NUM):
				if curr_status.group_batch[j] > 0:
					break
			time_stamp += compute_time(j, j+1, curr_status.group_batch[j], curr_status)
			wait_time += compute_time(j, j+1, curr_status.group_batch[j], curr_status) * (np.array(curr_status.group_batch).sum()+wait_job)
			if j == GROUP_NUM-1:
				curr_status.group_batch[j] = 0
			else:
				curr_status.group_batch[j+1] += curr_status.group_batch[j]
				curr_status.group_batch[j] = 0

	return wait_time

def no_batching(curr_status, new_req_seq):
	time_stamp = 0.0
	wait_time = 0.0
	wait_job = 0
	i = 0
	while i < NUM_NEW_REQUEST or np.array(curr_status.group_batch).sum() != 0:
		if (wait_job > 0 and np.array(curr_status.group_batch).sum() < MEMORY_SIZE) or (i < NUM_NEW_REQUEST and new_req_seq[i] <= time_stamp):
			if np.array(curr_status.group_batch).sum() >= MEMORY_SIZE:
				wait_job += 1
			elif wait_job > 0:
				curr_status.group_batch[0] += 1
				wait_job -= 1
			else:
				curr_status.group_batch[0] += 1
			i += 1
		elif np.array(curr_status.group_batch).sum() == 0:
			time_stamp = max(time_stamp, new_req_seq[i])
		else:
			for j in range(GROUP_NUM-1, -1, -1):
				if curr_status.group_batch[j] > 0:
					break
			time_stamp += compute_time(j, j+1, 1, curr_status)
			wait_time += compute_time(j, j+1, 1, curr_status) * (np.array(curr_status.group_batch).sum()+wait_job)
			if j == GROUP_NUM-1:
				curr_status.group_batch[j] -= 1
			else:
				curr_status.group_batch[j+1] += 1
				curr_status.group_batch[j] -= 1

	return wait_time

if __name__ == '__main__':
	curr_status = System_Status()
	read_layer(curr_status, "vgg16_titanx_default_pred.txt")
	curr_status.group_batch[0] = 3
	curr_status.group_batch[1] = 1
	curr_status.group_batch[2] = 5
	curr_status.group_batch[3] = 0
	curr_status.group_batch[4] = 2
	f = open('request.txt','r')
	new_req_seq = []
	for i in f.readline().split():
	    new_req_seq.append(float(i))
	f.close()
	wait_time = always_batching(curr_status, new_req_seq)
	print(wait_time)




