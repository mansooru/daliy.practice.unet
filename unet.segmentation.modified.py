import tensorflow as tf
import numpy as np,sys,os
from sklearn.utils import shuffle
from scipy.ndimage import imread
from scipy.misc import imresize
import matplotlib.pyplot as plt
from utils import read_mat

np.random.seed(678)
tf.set_random_seed(1400)

# load DataSet
train_X, train_label=read_mat('./data/data_set.mat', True)
test_X, test_label=read_mat('./data/test_set.mat', True)
train_images = np.expand_dims(train_X[0:1400,:,:],axis=3)
train_labels = np.expand_dims(train_label[0:1400,:,:],axis=3)
train_images = (train_images - train_images.min()) / (train_images.max() - train_images.min())
train_labels = (train_labels - train_labels.min()) / (train_labels.max() - train_labels.min())

test_images = np.expand_dims(test_X[0:400,:,:],axis=3)
test_labels = np.expand_dims(test_label[0:400,:,:],axis=3)
test_images = (test_images - test_images.min()) / (test_images.max() - test_images.min())
test_labels = (test_labels - test_labels.min()) / (test_labels.max() - test_labels.min())

def tf_relu(x): return tf.nn.relu(x)
def d_tf_relu(s): return tf.cast(tf.greater(s,0),dtype=tf.float32)
def tf_softmax(x): return tf.nn.softmax(x)
def np_sigmoid(x): 1/(1 + np.exp(-1 *x))

# --- make class ---
class conlayer_left():

    def __init__(self,ker,in_c,out_c):
        self.w = tf.Variable(tf.random_normal([ker,ker,in_c,out_c],stddev=0.05))

    def feedforward(self,input,stride=1,dilate=1):
        self.input  = input
        self.layer  = tf.nn.conv2d(input,self.w,strides = [1,stride,stride,1],padding='SAME')
        self.layerA = tf_relu(self.layer)
        return self.layerA

class conlayer_right():

    def __init__(self,ker,in_c,out_c):
        self.w = tf.Variable(tf.random_normal([ker,ker,in_c,out_c],stddev=0.05))

    def feedforward(self,input,stride=1,dilate=1,output=1):
        self.input  = input

        current_shape_size = input.shape

        self.layer = tf.nn.conv2d_transpose(input,self.w,
        output_shape=[batch_size] + [int(current_shape_size[1].value*2),int(current_shape_size[2].value*2),int(current_shape_size[3].value/2)],strides=[1,2,2,1],padding='SAME')
        self.layerA = tf_relu(self.layer)
        return self.layerA

# --- hyper ---
num_epoch = 2000
init_lr = 0.0001
batch_size = 20

# --- make layer ---
# left
l1_1 = conlayer_left(3,1,64)
l1_2 = conlayer_left(3,64,64)
l1_3 = conlayer_left(3,64,64)

l2_1 = conlayer_left(3,64,128)
l2_2 = conlayer_left(3,128,128)
l2_3 = conlayer_left(3,128,128)

l3_1 = conlayer_left(3,128,256)
l3_2 = conlayer_left(3,256,256)
l3_3 = conlayer_left(3,256,128)

# right
l4_1 = conlayer_right(3,128,256)
l4_2 = conlayer_left(3,128,128)
l4_3 = conlayer_left(3,128,64)

l5_1 = conlayer_right(3,64,128)
l5_2 = conlayer_left(3,64,64)
l5_3 = conlayer_left(3,64,64)

l6_final = conlayer_left(3,64,1)

# ---- make graph ----
x = tf.placeholder(shape=[None,256,256,1],dtype=tf.float32)
y = tf.placeholder(shape=[None,256,256,1],dtype=tf.float32)

layer1_1 = l1_1.feedforward(x)
layer1_2 = l1_2.feedforward(layer1_1)
layer1_3 = l1_3.feedforward(layer1_2)

layer2_Input = tf.nn.max_pool(layer1_3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer2_1 = l2_1.feedforward(layer2_Input)
layer2_2 = l2_2.feedforward(layer2_1)
layer2_3 = l2_3.feedforward(layer2_2)

layer3_Input = tf.nn.max_pool(layer2_3,ksize=[1,2,2,1],strides=[1,2,2,1],padding='VALID')
layer3_1 = l3_1.feedforward(layer3_Input)
layer3_2 = l3_2.feedforward(layer3_1)
layer3_3 = l3_3.feedforward(layer3_2)

layer4_Input = tf.concat([layer3_3,layer3_Input],axis=3)
layer4_1 = l4_1.feedforward(layer4_Input)
layer4_2 = l4_2.feedforward(layer4_1)
layer4_3 = l4_3.feedforward(layer4_2)

layer5_Input = tf.concat([layer4_3,layer2_Input],axis=3)
layer5_1 = l5_1.feedforward(layer5_Input)
layer5_2 = l5_2.feedforward(layer5_1)
layer5_3 = l5_3.feedforward(layer5_2)

layer6 = l6_final.feedforward(layer5_3)

cost = tf.reduce_mean(tf.square(layer6-y))
auto_train = tf.train.AdamOptimizer(learning_rate=init_lr).minimize(cost)

# --- start session ---
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for iter in range(num_epoch):
    index = np.arange(train_images.shape[0])
    np.random.shuffle(index)
    trX1 = train_images[index]
    trX2= train_labels[index]
    # train
    for current_batch_index in range(0,len(train_images),batch_size):
        current_batch = trX1[current_batch_index:current_batch_index+batch_size,:,:,:]
        current_label = trX2[current_batch_index:current_batch_index+batch_size,:,:,:]
        sess_results = sess.run([cost,auto_train],feed_dict={x:current_batch,y:current_label})
        print(" Cost:  %.30f" % sess_results[0], ' Iter: ', current_batch_index, '   ', end='\r' )
    print('\n-----------------------')

for data_index in range(0,len(train_images),batch_size):
    current_batch = train_images[data_index:data_index+batch_size,:,:,:]
    current_label = train_labels[data_index:data_index+batch_size,:,:,:]
    sess_results = sess.run(layer6,feed_dict={x:current_batch})

    plt.figure()
    plt.imshow(np.squeeze(current_batch[0,:,:,:]),cmap='gray')
    plt.axis('off')
    plt.title(str(data_index)+"a_Original Image")
    plt.savefig('./gif/'+str(data_index)+"a_Original_Image.png")

    plt.figure()
    plt.imshow(np.squeeze(current_label[0,:,:,:]),cmap='gray')
    plt.axis('off')
    plt.title(str(data_index)+"b_Original Mask")
    plt.savefig('./gif/'+str(data_index)+"b_Original_Mask.png")

    plt.figure()
    plt.imshow(np.squeeze(sess_results[0,:,:,:]),cmap='gray')
    plt.axis('off')
    plt.title(str(data_index)+"c_Generated Mask")
    plt.savefig('./gif/'+str(data_index)+"c_Generated_Mask.png")

    plt.figure()
    plt.imshow(np.multiply(np.squeeze(current_batch[0,:,:,:]),np.squeeze(current_label[0,:,:,:])),cmap='gray')
    plt.axis('off')
    plt.title(str(data_index)+"d_Original Image Overlay")
    plt.savefig('./gif/'+str(data_index)+"d_Original_Image_Overlay.png")

    plt.figure()
    plt.imshow(np.multiply(np.squeeze(current_batch[0,:,:,:]),np.squeeze(sess_results[0,:,:,:])),cmap='gray')
    plt.axis('off')
    plt.title(str(data_index)+"e_Generated Image Overlay")
    plt.savefig('./gif/'+str(data_index)+"e_Generated_Image_Overlay.png")

    plt.close('all')


for data_index in range(0,len(test_images),batch_size):
    current_batch = test_images[data_index:data_index+batch_size,:,:,:]
    current_label = test_labels[data_index:data_index+batch_size,:,:,:]
    sess_results = sess.run(layer6,feed_dict={x:current_batch})

    plt.figure()
    plt.imshow(np.squeeze(current_batch[0,:,:,:]),cmap='gray')
    plt.axis('off')
    plt.title(str(data_index)+"a_Original Image")
    plt.savefig('./gif/test_'+str(data_index)+"a_Original_Image.png")

    plt.figure()
    plt.imshow(np.squeeze(current_label[0,:,:,:]),cmap='gray')
    plt.axis('off')
    plt.title(str(data_index)+"b_Original Mask")
    plt.savefig('./gif/test_'+str(data_index)+"b_Original_Mask.png")

    plt.figure()
    plt.imshow(np.squeeze(sess_results[0,:,:,:]),cmap='gray')
    plt.axis('off')
    plt.title(str(data_index)+"c_Generated Mask")
    plt.savefig('./gif/test_'+str(data_index)+"c_Generated_Mask.png")

    plt.figure()
    plt.imshow(np.multiply(np.squeeze(current_batch[0,:,:,:]),np.squeeze(current_label[0,:,:,:])),cmap='gray')
    plt.axis('off')
    plt.title(str(data_index)+"d_Original Image Overlay")
    plt.savefig('./gif/test_'+str(data_index)+"d_Original_Image_Overlay.png")

    plt.figure()
    plt.imshow(np.multiply(np.squeeze(current_batch[0,:,:,:]),np.squeeze(sess_results[0,:,:,:])),cmap='gray')
    plt.axis('off')
    plt.title(str(data_index)+"e_Generated Image Overlay")
    plt.savefig('./gif/test_'+str(data_index)+"e_Generated_Image_Overlay.png")

    plt.close('all')


# -- end code --
