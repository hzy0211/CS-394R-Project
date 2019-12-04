from utils import *
from Request import Request
class System_Status:
	def __init__(self):
		self.system_label=0
		self.num_ini_request=0
		self.num_layer=0
		self.cur_layer=0
		self.num_shared_layers=0
		self.num_shared_layer_groups=0
		self.cur_batch = [0]*max(LAYER_SIZE_VGG,LAYER_SIZE_RES)
		self.group_batch=[0]*max(GROUP_NUM_VGG,GROUP_NUM_RES)
		self.shared_layer_group_idx=[0]*10
		self.num_old_request=0
		self.num_new_request=0
		self.ini_end_idx=0
		self.new_end_idx=0
		self.num_out=0
		self.re_evaluate=False
		self.total_timestamp=0.0
		self.batch_matrix = []
		self.batch_matrix_pred = []
		self.group_batch_matrix = []
		self.group_batch_matrix_pred = []
		for i in range(max(BATCH_SIZE_VGG,BATCH_SIZE_RES)):
			self.batch_matrix.append([0.0]*max(LAYER_SIZE_VGG,LAYER_SIZE_RES))
			self.batch_matrix_pred.append([0.0]*max(LAYER_SIZE_VGG,LAYER_SIZE_RES))
			self.group_batch_matrix.append([0.0]*max(GROUP_NUM_VGG,GROUP_NUM_RES))
			self.group_batch_matrix_pred.append([0.0]*max(GROUP_NUM_VGG,GROUP_NUM_RES))
			
		self.ini_req_list=[]
		self.new_req_list=[]
		for i in range(QUEUE_SIZE):
			self.ini_req_list.append(Request())
			self.new_req_list.append(Request())

		
'''
struct System_Status{
    int system_label;
	int num_ini_request;  //the number of initial requests in the whole system
	int num_layer;  //the number of layers with jobs
	int cur_layer;  //current running layer
    int num_shared_layers;
    int num_shared_layer_groups; //the number of layer groups of shared layers
	int cur_batch[MAX(LAYER_SIZE_VGG,LAYER_SIZE_RES)];  //store the number of requests for each layer
	int group_batch[MAX(GROUP_NUM_VGG,GROUP_NUM_RES)];  //store the number of requests for each layer
    int shared_layer_group_idx[10];
	int num_old_request;
    int num_new_request;
	int ini_end_idx;  //the index of old completed requests
	int new_end_idx;  //the index of new completed requests
	int num_out;  //the number of completed requests
	bool re_evaluate; //whether the queue size for shared and exclusive layers have changed
	//double wait_time[1000000];  //total waiting time for each combination
	double total_timestamp;  //define the current timestamp
        //store the running time for each layer and different batch size
	double batch_matrix[MAX(BATCH_SIZE_VGG,BATCH_SIZE_RES)][MAX(LAYER_SIZE_VGG,LAYER_SIZE_RES)];  
    double batch_matrix_pred[MAX(BATCH_SIZE_VGG,BATCH_SIZE_RES)][MAX(LAYER_SIZE_VGG,LAYER_SIZE_RES)];
	double group_batch_matrix[MAX(BATCH_SIZE_VGG,BATCH_SIZE_RES)][MAX(LAYER_SIZE_VGG,LAYER_SIZE_RES)];
    double group_batch_matrix_pred[MAX(BATCH_SIZE_VGG,BATCH_SIZE_RES)][MAX(LAYER_SIZE_VGG,LAYER_SIZE_RES)];
	double req_matrix[NUM_TESTCASE][NUM_NEW_REQUEST];  //store all new requests
	//int bottleneck[NUM_BOTTLENECK] = {2,7,12,14,17,19,21,31};  //vgg16-pytorch
	//int bottleneck[NUM_BOTTLENECK] = {7,14,19,31};  //vgg16-pytorch
	//int bottleneck[NUM_BOTTLENECK] = {59,103,107,112,119};  //resnet-pytorch
	//int bottleneck[NUM_BOTTLENECK]; // = {103,112};  //resnet-pytorch

	struct Request ini_req_list[QUEUE_SIZE];
	struct Request new_req_list[QUEUE_SIZE];
};
'''
