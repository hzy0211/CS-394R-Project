from utils import *

class Read_Layer:
        def __init__(self, curr_status_list_t, model_parmt_t):
                self.curr_status_list = curr_status_list_t
                self.model_parmt = model_parmt_t

        def read_layer_vgg(self,curr_status,batch_size,layer_size,group_num_t,data_file):
                #print("batch size: ",batch_size)
                #print("num_shared_layers: ",curr_status.num_shared_layers)
                #print("group_num_t: ",group_num_t, "group_num_shared: ",GROUP_NUM_SHARED)
                curr_sum = 0.0
                fp = open(data_file,"r")
                for i in range(batch_size):
                        curr_layer = fp.readline().strip()
                        curr_layer = curr_layer.split()
                        for j in range(layer_size):
                                curr_status.batch_matrix[i][j] = round(float(curr_layer[j]),6)
                        if i==0:
                                curr_sum += curr_status.batch_matrix[i][j]

                fp.close()
                #print('first batch: ',curr_status.batch_matrix[0][:])
                '''
                temp = ''
                for i in range(batch_size):
                        temp = temp + ' '+str(curr_status.batch_matrix[i][0])
                print("first layer: ", temp)
                '''
                #print("first layer: ",curr_status.batch_matrix[:][0])

                group_num = GROUP_NUM_SHARED
                j = 0
                curr_sum = 0.0
                for i in range(curr_status.num_shared_layers):
                	curr_sum += curr_status.batch_matrix[0][i]

                for k in range(batch_size):
                	curr_status.group_batch_matrix[k][j] = 0.0

                for i in range(curr_status.num_shared_layers):
                        assert (j<group_num_t)
                        for k in range(batch_size):
                                curr_status.group_batch_matrix[k][j] += curr_status.batch_matrix[k][i]

                        if (curr_status.group_batch_matrix[0][j]>=curr_sum/(1.0*group_num)) or (i==curr_status.num_shared_layers-1):
                                curr_status.shared_layer_group_idx[j]=i
                                curr_sum -= curr_status.group_batch_matrix[0][j]
                                group_num -= 1
                                if abs(curr_sum)<1e-8:
                                        curr_sum=0.0
                                #print("curr_sum: ",curr_sum,"group_num: ",group_num,"i: ",i, "num_shared_layers: ", curr_status.num_shared_layers)

                                assert (group_num>=0)
                                assert (curr_sum>=0)
                                j += 1
                                for k in range(batch_size):
                                        curr_status.group_batch_matrix[k][j]=0.0
                                        
                curr_status.num_shared_layer_groups = j
                group_num = group_num_t - GROUP_NUM_SHARED
                #print("group_num: ",group_num)
                assert (group_num>0)
                #print("shared_layer_group_idx: ",curr_status.shared_layer_group_idx)

                #print("num_shared_layers: ",curr_status.num_shared_layers)
                curr_sum = 0.0
                for i in range(curr_status.num_shared_layers,layer_size):
                        curr_sum += curr_status.batch_matrix[0][i]

                print("python curr_sum: ",curr_sum)
                for k in range(batch_size):
                        curr_status.group_batch_matrix[k][j] = 0.0

                for i in range(curr_status.num_shared_layers,layer_size):
                        for k in range(batch_size):
                                curr_status.group_batch_matrix[k][j] += curr_status.batch_matrix[k][i]
                        #divide layers into gruops such that each group has roughly equal running time
                        if (curr_status.group_batch_matrix[0][j]>=curr_sum/(1.0*group_num)):
                                #when reaching the last shared layer, we form a group
                                curr_sum -= curr_status.group_batch_matrix[0][j]
                                group_num -= 1
                                if abs(curr_sum)<1e-08:
                                        curr_sum = 0.0
                                        
                                assert (group_num>=0)
                                assert(curr_sum>=0)
                                j += 1
                                if j<GROUP_NUM_VGG:
                                        for k in range(batch_size):
                                                curr_status.group_batch_matrix[k][j] = 0.0

        def read_layer_res(self,curr_status, curr_status_other, batch_size, layer_size, group_num_t, data_file):
                curr_sum = 0
                fp = open(data_file,"r")
                for i in range(batch_size):
                        curr_layer = fp.readline().strip()
                        curr_layer = curr_layer.split()
                        #print('layer_size: ',layer_size, 'read_data: ',len(curr_layer))
                        for j in range(layer_size):
                                curr_status.batch_matrix[i][j] = round(float(curr_layer[j]),6)
                        if i==0:
                                curr_sum += curr_status.batch_matrix[i][j]

                fp.close()
                j=0

                for k in range(batch_size):
                        curr_status.group_batch_matrix[k][j] = 0.0

                for i in range(curr_status.num_shared_layers):
                        for k in range(batch_size):
                                curr_status.group_batch_matrix[k][j] += curr_status.batch_matrix[k][i]
                        if curr_status_other.shared_layer_group_idx[j]==i:
                                j += 1
                                for k in range(batch_size):
                                        curr_status.group_batch_matrix[k][j] = 0.0
                    
                curr_status.num_shared_layer_groups = j
                group_num = group_num_t - GROUP_NUM_SHARED
                assert (group_num>0)

                curr_sum = 0.0
                for i in range(curr_status.num_shared_layers,layer_size):
                        curr_sum += curr_status.batch_matrix[0][i]

                for k in range(batch_size):
                        curr_status.group_batch_matrix[k][j] = 0.0

                for i in range(curr_status.num_shared_layers,layer_size):
                        for k in range(batch_size):
                                curr_status.group_batch_matrix[k][j] += curr_status.batch_matrix[k][i]

                        if curr_status.group_batch_matrix[0][j] >= curr_sum/(1.0*group_num):
                                #when reaching the last shared layer, we form a group
                                if i==curr_status.num_shared_layers:
                                        #set the number of shared layer groups
                                        curr_status.num_shared_layer_groups = j

                                curr_sum -= curr_status.group_batch_matrix[0][j]
                                group_num -= 1
                                if abs(curr_sum)<1e-08:
                                        curr_sum = 0.0
                                assert(curr_sum>=0)
                                assert(group_num>=0)
                                j += 1
                                if j<GROUP_NUM_RES:
                                        for k in range(batch_size):
                                                curr_status.group_batch_matrix[k][j] = 0.0

        def read_layer_multi_models(self):
                self.read_layer_vgg(self.curr_status_list[0], self.model_parmt.batch_size_list[0],\
                               self.model_parmt.layer_size_list[0], self.model_parmt.group_num_list[0],\
                               self.model_parmt.path_list[0])

                self.read_layer_res(self.curr_status_list[1], self.curr_status_list[0],\
                               self.model_parmt.batch_size_list[1], self.model_parmt.layer_size_list[1],\
                               self.model_parmt.group_num_list[1],self.model_parmt.path_list[1])
