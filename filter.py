## Capture the possible effect of each layer, max pool not included
# Usage: 
# job=filt_to_image(model) # establish class and list convolution layers
# job.get_equ_fil(n) #calculate get_equ_filter for each possible layers (~layer n)
# job.plot_eqfil(n,n_samples) # plot the effect of equivalent of n th layer 
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
class filt_to_image():
  
  def __init__(self,model):

    self.model=model
    self.conv_layers=[layer.name for layer in self.model.layers if ("conv" in layer.name)]
    self.nlayers=len(self.conv_layers)
    self.equ_fil=[0]*self.nlayers
    self.get_equ_fil(self.nlayers-1);
  def get_equ_fil(self,rng,pad_acc=0):
    fil=self.model.get_layer(self.conv_layers[rng]).get_weights()[0]
    PAD=(fil.shape[0]-1)//2
    if rng==0:    
      img=tf.random.normal([1,fil.shape[0]+2*pad_acc,fil.shape[1]+2*pad_acc,fil.shape[2]])
    else:
      img=self.get_equ_fil(rng-1,pad_acc+PAD)
    self.equ_fil[rng]=tf.nn.conv2d(img, fil, strides= [1, 1, 1, 1], padding="SAME")
    return self.equ_fil[rng]
  def plot_eqfil(self,rng=0,n_samples=25):

    x_a=np.ceil(np.sqrt(n_samples))
    y_a=np.floor(np.sqrt(n_samples))
    fig = plt.figure(figsize=(10,10))
    fig.subplots_adjust(hspace=0.02,wspace=0.01,
                    left=0,right=1,bottom=0, top=1)
    ## original picture
    total_samples=self.equ_fil[rng].shape[3]
    order=np.linspace(1,total_samples-1,total_samples,dtype="int")
    random.shuffle(order)
    i = 1
    for sample in order[0:n_samples]:
        ax = fig.add_subplot(x_a, y_a, i,xticks=[],yticks=[]) 
        ax.imshow(np.squeeze(self.equ_fil[rng][:,:,:,sample]))
        ax.set_title(f"sample{sample}")
        i += 1
    plt.show()
