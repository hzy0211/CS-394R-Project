from utils import *
from Request import Request
from Model_Parameters import Model_Parameters
from System_Status import System_Status
from Read_Layer import Read_Layer

class Schedule:
    def __init__(self):
        self.schedule_size = 0
        self.schedule_order = [0]*1000

class Request_Info:
    def __init__(self):
        self.model = 0
        self.layer = 0


class dp_scheduler:
    def __init__(self):
        group_num_list = [GROUP_NUM_VGG, GROUP_NUM_FCN]
        memory_size_list = [MEMORY_SIZE_VGG, MEMORY_SIZE_FCN]
        batch_size_list = [BATCH_SIZE_VGG, BATCH_SIZE_FCN]
        layer_size_list = [LAYER_SIZE_VGG, LAYER_SIZE_FCN]
        path_list = ["fcn_vgg16_titanx_default_pred.txt", "vgg16_titanx_default_pred.txt"]

        self.curr_status_list = []

        self.model_parmt = Model_Parameters(NUM_MODEL)

        for i in range(NUM_MODEL):
            self.curr_status_list.append(System_Status())
            self.curr_status_list[i].system_label = i
            self.model_parmt.group_num_list[i] = group_num_list[i]
            self.model_parmt.memory_size_list[i] = memory_size_list[i]
            self.model_parmt.batch_size_list[i] = batch_size_list[i]
            self.model_parmt.layer_size_list[i] = layer_size_list[i]
            self.model_parmt.path_list[i] = path_list[i][:]

        self.layer_reader = Read_Layer(self.curr_status_list, self.model_parmt)
        for n in range(NUM_MODEL):
            self.set_shared_layers(self.curr_status_list[n], NUM_SHARED_LAYERS, GROUP_NUM_SHARED)

        self.layer_reader.read_layer_multi_models()
        self.opt_schedule = Schedule()


    def set_shared_layers(self,curr_status, num_shared_layers_t, num_shared_layer_groups_t):
        curr_status.num_shared_layers = num_shared_layers_t
        curr_status.num_shared_layer_groups = num_shared_layer_groups_t


    def update_status(self,curr_status, curr_status_other, group_batch, group_batch_other, curr_layer, num_batch_curr, num_batch_other):
        group_num_curr = 0
        group_num_other = 0

        any_req_finished = False

        if curr_status.system_label==0:
            group_num_curr = GROUP_NUM_VGG
            group_num_other = GROUP_NUM_RES
        else:
            group_num_curr = GROUP_NUM_RES
            group_num_other = GROUP_NUM_VGG

        #update the group batch size for the current model
        if num_batch_curr>0:
            if (curr_layer+1==curr_status.num_shared_layer_groups) or (curr_layer+1==group_num_curr):
                curr_status.re_evaluate=True
            if curr_layer+1<group_num_curr:
                group_batch[curr_layer] -= num_batch_curr
                group_batch[curr_layer+1] += num_batch_curr
            else:
                group_batch[curr_layer] -= num_batch_curr
                curr_status.num_out = num_batch_curr
                #some requests are finished
                any_req_finished = True

        assert (group_batch[curr_layer]>=0)

        #update the group batch size for the other model
        if num_batch_other>0:
            if curr_layer+1==curr_status_other.num_shared_layer_groups:
                curr_status_other.re_evaluate = True
            assert(curr_layer+1<group_num_other)
            group_batch_other[curr_layer] -= num_batch_other
            group_batch_other[curr_layer+1] += num_batch_other
        assert (group_batch_other[curr_layer]>=0)
        return any_req_finished


    #get the start layer idx of a given schedule
    def get_start_layer(self, curr_status, curr_status_other, group_batch, group_batch_other, schedule):
        start_layer = 0
        curr_req_idx = 0
        group_num = 0
        if curr_status.system_label==0:
            group_num = GROUP_NUM_VGG
        else:
            group_num = GROUP_NUM_RES

        #get the corresponding layer idx of the current schedule
        while (curr_req_idx<=schedule) and (start_layer<group_num):
            if start_layer<curr_status.num_shared_layer_groups:
                curr_req_idx += group_batch[start_layer]
                curr_req_idx += group_batch_other[start_layer]
            else:
                curr_req_idx += group_batch[start_layer]

            if curr_req_idx<=schedule:
                start_layer += 1
            else:
                break
        return start_layer

    def apply_a_schedule(self, curr_status, curr_status_other, group_batch, group_batch_other, schedule, system_time):
        start_layer = 0
        curr_layer = 0
        group_num = 0
        wait_time = 0.0

        if curr_status.system_label==0:
            group_num = GROUP_NUM_VGG
        else:
            group_num = GROUP_NUM_RES

        start_layer = self.get_start_layer(curr_status, curr_status_other, group_batch, group_batch_other, schedule)
        req_idx = 0
        for i in range(start_layer):
            if i<curr_status.num_shared_layer_groups:
                req_idx += group_batch_other[i]
                req_idx += group_batch[i]
            else:
                req_idx += group_batch[i]

        num_batch_curr = 0
        num_batch_other = 0
        curr_layer = start_layer
        while (num_batch_curr+num_batch_other==0) and (curr_layer<group_num):
            if curr_layer<curr_status.num_shared_layer_groups:
                if req_idx+group_batch_other[curr_layer]>schedule:
                    num_batch_curr = group_batch[curr_layer]
                    num_batch_other = group_batch_other[curr_layer]-(schedule-req_idx)
                    assert(num_batch_curr>=0)
                    assert(num_batch_other>=0)
                else:
                    num_batch_curr = group_batch[curr_layer]-(schedule-(req_idx+group_batch_other[curr_layer]))
                    num_batch_other = 0
                    assert (num_batch_curr>=0)
            else:
                num_batch_curr = group_batch[curr_layer]-(schedule-req_idx)
                assert (num_batch_curr>=0)

            if num_batch_curr+num_batch_other>0:
                break
            curr_layer += 1
            
        if num_batch_curr+num_batch_other==0:
            return True
        else:
            wait_time = self.compute_time_layer(curr_layer, num_batch_curr+num_batch_other, curr_status)
            if system_time[0]!=None:
                system_time[0] += wait_time

            self.update_status(curr_status, curr_status_other, group_batch, group_batch_other, curr_layer, num_batch_curr, num_batch_other)
            
        return False

    #compute time for running a batch for one layer
    def compute_time_layer(self, curr_layer, batch_size, curr_status):
        time = curr_status.group_batch_matrix[batch_size-1][curr_layer]
        return time

    #calculate the wait time for a given schedule
    def get_wait_time_for_a_schedule(self, req_seq, req_availability, num_req, curr_status, curr_status_other,\
                                     schedule, curr_system_time, updated_system_time):
        curr_wait_time = 0
        curr_layer = 0
        group_num = 0
        memory_size = 0
        if curr_status.system_label==0:
            group_num = GROUP_NUM_VGG
            memory_size = MEMORY_SIZE_VGG
        else:
            group_num = GROUP_NUM_RES
            memory_size = MEMORY_SIZE_RES

        #start from the layer indicated by the schedule
        curr_layer = req_seq[schedule].layer

        req_idx = schedule
        curr_req_num = 0
        curr_req_num_other = 0
        while curr_layer<group_num:
            while (req_idx<num_req) and (req_seq[req_idx].layer<=curr_layer):
                assert (req_idx<num_req)
                if req_availability[req_idx]:
                    curr_req_num += 1
                if (req_availability[req_idx]) and (req_seq[req_idx].model==curr_status_other.system_label):
                    curr_req_num_other += 1
                req_idx += 1

            if curr_layer==curr_status.num_shared_layer_groups:
                curr_req_num -= curr_req_num_other
                assert (curr_req_num>=0)

            #set wait time to INF if the batch size is greater than the memory bound
            if curr_req_num>memory_size:
                curr_wait_time = MAX_FLOAT
                break
            #run the current batch for one layer
            assert (curr_layer<group_num)
            #print("curr_req_num: ",curr_req_num, "memory_size: ",memory_size)
            assert (curr_req_num<=memory_size)
            if curr_req_num>0:
                #print("running time: ", curr_status.group_batch_matrix[curr_rseq_num-1][curr_layer],\
                #"curr_layer: ",curr_layer, "curr_req_num: ",curr_req_num)
                curr_wait_time += curr_status.group_batch_matrix[curr_req_num-1][curr_layer]
                
            #the current batch reaches the next layer
            curr_layer += 1
        if curr_wait_time!=MAX_FLOAT:
            updated_system_time[0] = curr_system_time[0]+curr_wait_time
            
        #all requests arrival time is 0, before they are scheduled,
        #they have already been waiting for the value of curr_system_time
        return (curr_system_time[0]+curr_wait_time)*curr_req_num

    def update_req_availability(self, req_availability, schedule, num_req):
        for i in range(schedule, num_req):
            req_availability[i] = 0


    #get the optimal schedule by checking layer boundaries
    def get_optimal_schedule(self, opt_schedule, group_batch, group_batch_other,\
                             curr_status, curr_status_other, system_time):

        #print("group_batch: ",group_batch)
        #print("group_batch_other: ",group_batch_other)
        num_req = 0
        wait_time = 0.0
        group_num = GROUP_NUM_VGG
        memory_size = MEMORY_SIZE_VGG
        if curr_status.system_label==0:
            group_num = GROUP_NUM_VGG
            memory_size = MEMORY_SIZE_VGG
        else:
            group_num = GROUP_NUM_RES
            memory_size = MEMORY_SIZE_RES

        for i in range(curr_status_other.num_shared_layer_groups):
            num_req += group_batch_other[i]

        for i in range(group_num):
            num_req += group_batch[i]

        if num_req==0:
            opt_schedule.schedule_size = 1
            opt_schedule.schedule_order[0] = 0
            return 0
        
        num_decision_points = 0
        layer_idx = [0]*50
        req_idx = [0]*50
        curr_req_idx = 0
        #get the decision points
        for i in range(curr_status.num_shared_layer_groups):
            curr_batch_size = group_batch_other[i]
            while curr_batch_size>0:
                req_idx[num_decision_points] = curr_req_idx
                layer_idx[num_decision_points] = i
                if curr_batch_size%memory_size>0:
                    curr_req_idx += (curr_batch_size%memory_size)
                    curr_batch_size -= (curr_batch_size%memory_size)
                else:
                    curr_req_idx += min(curr_batch_size, memory_size)
                    curr_batch_size = max(curr_batch_size-memory_size,0)
                num_decision_points += 1
                
            curr_batch_size = group_batch[i]
            while curr_batch_size>0:
                req_idx[num_decision_points] = curr_req_idx
                layer_idx[num_decision_points] = i
                if curr_batch_size%memory_size>0:
                    curr_req_idx += (curr_batch_size%memory_size)
                    curr_batch_size -= (curr_batch_size%memory_size)
                else:
                    curr_req_idx += min(curr_batch_size, memory_size)
                    curr_batch_size = max(curr_batch_size-memory-size,0)
                    
                num_decision_points += 1
        for i in range(curr_status.num_shared_layer_groups, group_num):
            curr_batch_size = group_batch[i]
            while curr_batch_size>0:
                req_idx[num_decision_points] = curr_req_idx
                layer_idx[num_decision_points] = i
                if curr_batch_size%memory_size>0:
                    curr_req_idx += (curr_batch_size%memory_size)
                    curr_batch_size -= (curr_batch_size%memory_size)
                else:
                    curr_req_idx += min(curr_batch_size, memory_size)
                    curr_batch_size = max(curr_batch_size-memory_size,0)
                    
                num_decision_points += 1

        #print("decision_points: ",req_idx)
        #generate the request sequence
        req_seq = []
        for i in range(num_req):
            req_seq.append(Request_Info())
        #req_available stores whether a request in the sequence has already finished or not
        req_available = [1]*num_req
        curr_req_idx = 0
        #the request sequence is ordered by arrival time
        for i in range(curr_status.num_shared_layer_groups):
            for j in range(group_batch_other[i]):
                req_seq[curr_req_idx].model = curr_status_other.system_label
                req_seq[curr_req_idx].layer = i
                curr_req_idx += 1
            for j in range(group_batch[i]):
                req_seq[curr_req_idx].model = curr_status.system_label
                req_seq[curr_req_idx].layer = i
                curr_req_idx += 1
        for i in range(curr_status.num_shared_layer_groups, group_num):
            for j in range(group_batch[i]):
                req_seq[curr_req_idx].model = curr_status.system_label
                req_seq[curr_req_idx].layer = i
                curr_req_idx += 1
        minTime = [0.0]*(num_decision_points+1)
        curr_system_time = [0.0]*(num_decision_points+1)
        curr_system_time[num_decision_points] = system_time[0]
        
        temp = 0.0
        #track the optimal splitting point for a given request
        track_opt = [0]*num_decision_points
        req_available_t = [0]*num_req
        temp_system_time = [system_time[0]]

        for i in range(num_decision_points-1,-1,-1):
            minTime[i] = MAX_FLOAT
            for j in range(i+1,num_decision_points+1):
                req_i = req_idx[i]
                req_j = req_idx[j]
                if j==num_decision_points:
                    req_j = num_req
                #check the optimal splitting point of the sequence after the request i
                req_available_t = req_available[:]

                #after applying the schedule [j], all requests [j,...] are finished
                self.update_req_availability(req_available_t, req_j, num_req)

                assert (req_i<=num_req)
                assert (req_j<=num_req)

                curr_system_time_t = [curr_system_time[j]]
                
                temp = self.get_wait_time_for_a_schedule(req_seq, req_available_t,num_req,curr_status,curr_status_other,req_i,\
                                                    curr_system_time_t,temp_system_time)+minTime[j]
                #print("i: ",i,"j: ",j,"temp: ",temp)
                curr_system_time[j] = curr_system_time_t[0]
                if (temp<=minTime[i]) and (temp>=0):
                    #the optimal splitting point gives the minimum wait time
                    minTime[i] = temp;
                    track_opt[i] = j

                    curr_system_time[i] = temp_system_time[0]

        optimal_schedule = [-1]*num_req
        schedule_idx = num_req-1
        optimal_schedule[schedule_idx] = 0
        #print(minTime)
        #print(track_opt)

        schedule_idx -= 1
        k = track_opt[0]

        while k<num_decision_points:
            #calculate the optimal schedule
            assert ((schedule_idx<num_req) and (schedule_idx>=0))
            optimal_schedule[schedule_idx] = req_idx[k]
            k = track_opt[k]
            schedule_idx -= 1
            
        schedule_idx += 1
        assert(schedule_idx<num_req)
        schedule_size = (num_req-1)-schedule_idx+1
        #make sure schedule_size is smaller than a pre-defined capacity 1000
        assert ((schedule_size<=1000) and (schedule_size>=0))
        opt_schedule.schedule_size = schedule_size
        assert(schedule_idx+schedule_size<=num_req)
        opt_schedule.schedule_order[0:schedule_size] = optimal_schedule[schedule_idx:schedule_idx+schedule_size]
        wait_time = minTime[0]

        #the current system time is the time point when all requests of the request sequence are finished
        system_time[0] = curr_system_time[0]

        return wait_time

    def get_scheduling_order_sequential_dp(self, opt_schedule):
        curr_status_vgg = self.curr_status_list[0]
        curr_status_res = self.curr_status_list[1]

        scheduling_order = 0
        system_time=0.0
        time1 = 0.0
        time2 = 0.0

        opt_schedule_vgg = Schedule()
        opt_schedule_res = Schedule()

        group_batch_vgg = curr_status_vgg.group_batch[:]
        group_batch_res = curr_status_res.group_batch[:]

        req_num_total = 0
        start_layer = 0

        req_num_vgg = 0
        for i in range(GROUP_NUM_VGG):
            req_num_vgg += group_batch_vgg[i]

        req_num_res = 0
        for i in range(GROUP_NUM_RES):
            req_num_res += group_batch_res[i]

        if req_num_vgg==0:
            system_time_t = [0.0]
            time2 = self.get_optimal_schedule(opt_schedule_res, group_batch_res,group_batch_vgg,\
                                         curr_status_res, curr_status_vgg, system_time_t)
            system_time = system_time_t[0]
            opt_schedule.schedule_size = opt_schedule_res.schedule_size
            opt_schedule.schedule_order = opt_schedule_res.schedule_order[:]
            return 1
        if req_num_res==0:
            system_time_t = [0.0]
            time1 = self.get_optimal_schedule(opt_schedule_vgg, group_batch_vgg, group_batch_res,\
                                         curr_status_vgg, curr_status_res, system_time_t)
            system_time = system_time_t[0]
            opt_schedule.schedule_size = opt_schedule_vgg.schedule_size
            opt_schedule.schedule_order = opt_schedule_vgg.schedule_order[:]
            return 0

        system_time = 0.0
        system_time_t = [system_time]
        time1 = self.get_optimal_schedule(opt_schedule_vgg, group_batch_vgg, group_batch_res,\
                                     curr_status_vgg, curr_status_res, system_time_t)
        system_time = system_time_t[0]
        #print("system_time: ", system_time)

        for i in range(opt_schedule_vgg.schedule_size):
            while True:
                if self.apply_a_schedule(curr_status_vgg, curr_status_res, group_batch_vgg,\
                                    group_batch_res, opt_schedule_vgg.schedule_order[i],[None]):
                    break
        system_time_t = [system_time]
        time1 += self.get_optimal_schedule(opt_schedule, group_batch_res, group_batch_vgg,\
                                      curr_status_res, curr_status_vgg, system_time_t)
        system_time = system_time_t[0]
        #print("system_time: ",system_time)
        for i in range(opt_schedule.schedule_size):
            while True:
                if self.apply_a_schedule(curr_status_res, curr_status_vgg, group_batch_res,\
                                    group_batch_vgg, opt_schedule.schedule_order[i], [None]):
                    break
        
        group_batch_vgg = curr_status_vgg.group_batch[:]
        group_batch_res = curr_status_res.group_batch[:]

        system_time = 0.0
        system_time_t = [system_time]
        time2 = self.get_optimal_schedule(opt_schedule_res, group_batch_res, group_batch_vgg,\
                                     curr_status_res, curr_status_vgg, system_time_t)
        system_time = system_time_t[0]
        #print("system_time: ",system_time)
        for i in range(opt_schedule_res.schedule_size):
            while True:
                if self.apply_a_schedule(curr_status_res, curr_status_vgg, group_batch_res,\
                                    group_batch_vgg, opt_schedule_res.schedule_order[i],[None]):
                    break

        system_time_t = [system_time]
        time2 += self.get_optimal_schedule(opt_schedule, group_batch_vgg, group_batch_res,\
                                      curr_status_vgg,curr_status_res, system_time_t)
        system_time = system_time_t[0]
        #print("system_time: ",system_time)
        for i in range(opt_schedule.schedule_size):
            while True:
                if self.apply_a_schedule(curr_status_vgg, curr_status_res, group_batch_vgg,\
                                    group_batch_res, opt_schedule.schedule_order[i], [None]):
                    break

        if time1<time2:
            scheduling_order = 0
            opt_schedule.schedule_size = opt_schedule_vgg.schedule_size
            opt_schedule.schedule_order = opt_schedule_vgg.schedule_order[:]
        else:
            scheduling_order = 1
            opt_schedule.schedule_size = opt_schedule_res.schedule_size
            opt_schedule.schedule_order = opt_schedule_res.schedule_order[:]
            
        return scheduling_order


    def call_decision_dp(self,batch_array):
        i = 0
        for n in range(NUM_MODEL):
            j = 0
            while j<self.model_parmt.group_num_list[n]:
                self.curr_status_list[n].group_batch[j] = batch_array[i]
                i += 1
                j += 1

        opt_schedule = Schedule()
        scheduling_order = self.get_scheduling_order_sequential_dp(opt_schedule)
        output = [0]*100

        output[0] = scheduling_order
        output[1:1+opt_schedule.schedule_size] = opt_schedule.schedule_order[0:opt_schedule.schedule_size]
        return output
