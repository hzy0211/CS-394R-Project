NUM_NEW_REQUEST=100  #the number of new requests
QUEUE_SIZE=10000
NUM_TESTCASE=5  #the number of cases used for testingWW
MAX_FLOAT=100000
NUM_USERS=10
FRAME_RATE=25
NUM_MODEL=2
LEFT_TO_RIGHT=True
RIGHT_TO_LEFT=False

GROUP_NUM_VGG=5 #add one extra group bcz of shared layers
GROUP_NUM_RES=5 #res=5 //add one extra group bcz of shared layers
GROUP_NUM_FCN=5 #res=5 //add one extra group bcz of shared layers
GROUP_NUM_SHARED=3
MEMORY_SIZE_VGG=90
MEMORY_SIZE_RES=90
MEMORY_SIZE_FCN=90
BATCH_SIZE_VGG=90
BATCH_SIZE_RES=90
BATCH_SIZE_FCN=90
LAYER_SIZE_VGG=38 #vgg16_only=38, resnet50_only=126
LAYER_SIZE_RES=63 #ssd_vgg16=63, ssd_resnet50=143
LAYER_SIZE_FCN=39 #fcn_vgg16=39
NUM_SHARED_LAYERS=26 #the number of shared vgg16 layers is 31, resnet50_shared=112

MEMORY_SIZE=MEMORY_SIZE_VGG  #maximum batch size for each layer
BATCH_SIZE=BATCH_SIZE_VGG  #maximum batch size for each layer
LAYER_SIZE = LAYER_SIZE_VGG  #total number of layers
GROUP_NUM = GROUP_NUM_VGG

