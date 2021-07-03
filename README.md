[image1]: assets/1.png 
[image2]: assets/2.png 
[image3]: assets/3.png 
[image4]: assets/4.png 
[image5]: assets/5.png 
[image6]: assets/6.png 
[image7]: assets/7.png 
[image8]: assets/8.png 
[image9]: assets/9.png 
[image10]: assets/10.png 
[image11]: assets/11.png 
[image12]: assets/12.png 
[image13]: assets/13.png 
[image14]: assets/14.png 
[image15]: assets/15.png 
[image16]: assets/16.png 
[image17]: assets/17.png 
[image18]: assets/18.png 
[image19]: assets/19.png 
[image20]: assets/20.png 
[image21]: assets/21.png 
[image22]: assets/22.png 
[image23]: assets/23.png 
[image24]: assets/24.png 
[image25]: assets/25.png 
[image26]: assets/26.png 
[image27]: assets/27.png 
[image28]: assets/28.png 
[image29]: assets/29.png 
[image30]: assets/31.png 


# Machine Vision

Overview of Machine Vision techniques.

## Content 
- [Convolutional Neural Networks](#conv_nero_net)
    - [Convolutional layers](#conv_layers)
    - [Rule of thumb for Convolutional layer implementation](#rule)
    - [Difference between MLPs and CNNs](#diff_mlp_cnn)
    - [Computational complexity](#comp_complex)
    - [Hyperparameters of Convolutional kernels](#conv_kernel)
        - [Kernel size](#kernel_size)
        - [Stride](#stride)
        - [Padding](#padding)
    - [Pooling Layers](#pooling_layers) 
        - [Construction](#pooling_construct)
        - [Aim of Convolutional and Pooling layers](#aim_conv_pool)
- [State of the art ConvNets](#state_of_art)
    - [LeNet](#lenet)
    - [AlexNet](#alexnet)
    - [VGGNet](#vggnet)
    - [ResNet](#resnet)
    - [Object detection](#object_detection)
        - [R-CNN](#r_cnn)
        - [Fast R-CNN](#fast_r_cnn)
        - [Faster R-CNN](#faster_r_cnn)
        - [YOLO](#yolo)
    - [Picture Segmentation](#pic_seg)
        - [Mask R-CNN](#mask_r_cnn)
        - [U-Net](#u_net)
    - [Transfer Learning](#trans_learn)
    - [Capsule Networks](#caps_net)    

- [Setup Instructions](#Setup_Instructions)
- [Acknowledgments](#Acknowledgments)
- [Further Links](#Further_Links)

# Convolutional Neural Networks <a id="conv_nero_net"></a>
- Example:

    ![image1]

    - Convolutional Neural Network, ConvNet or CNN is an artificial neural net with one or more Convolutional layers.
    - Such a net enables to analyse **space localized patterns**
    - Therefore such nets are the core of Computer Vision applications.

## Convolutional layers <a id="conv_layers"></a>
- Convolutional layers consist of groups of kernels.

    ![image3]

    - Each kernel is a small window which scans the input image from left upper corner to the right lower corner.
    - Each filter executes a convolution. 
    - Operation: **RELU(w * x + b**) 
    - **w * x** elemetwise multiplication of kernel weight matrix **w** and actual image segment **x** 
    - Kernels consist of weights which will be learned by backproopagation
    - Typical kernel sizes: 3x3 or 5x5
    - Example: 3x3 kernel --> parameters to learn: 9 weights  + 1 bias

- Jason Yosinski - [Visualize what kernels are doing](https://www.youtube.com/watch?v=AgkfIQ4IGaM)
- [Code and more info](https://yosinski.com/deepvis)

- In case of RGB: Each filter exists for each color channel

    ![image4]
    
    - In order to calculate the output you have to take the RELU triggered weighted sum over all color channels as shown above. 

- In case of multiple filter (here 3):

    ![image5]

    In case of early Conv-layers:
    - Detection of simple features like edges. Those filters operate like markers, each for a certain feature
    - One filter for horizontal edges
    - One filter for vertical edges 
    - One filter for color transitions
    - etc. 

    In case of deeper Conv-layers:
    - More complex combinations of simple features (like textures and certain shapes, forms)

    Near output:
    - Those Conv-layers could detect whole objects

- Convolutional layers:
    - help deep learning models to learn how to recognize features dependent on position  
    - they keep the 2D strucure of the image
    - they reduce number of parameters to learn

## Rule of thumb for Convolutional layer implementation <a id="rule"></a> 

- A larger amount of kernels allows to identify complexer features.
- More kernels, however, need more computational power.
- The optimal amount of kernels can vary from layer to layer. Often it is better to implement more kernels in deeper layers than in early layers.
- Try to keep computational power as low as possible. Use the lowest possible amount of kernels. Always try to make the model only as complex as needed. 

## Difference between MLPs and CNNs <a id="diff_mlp_cnn"></a>
| Multi-Layer-Perceptrons (MLPs)| ConvNet |
|--- |--- |
| only fully connected layers| local connected layers|
| densly connected | sparsely connected |
| only vectors as inputs | also matrices as input possible |
| possibly millions of parameters| parameter amount can be reduced |
| possibly overfitting | less susceptible to overfitting|

## Computational complexity <a id="comp_complex"></a>
| Multi-Layer-Perceptrons (MLPs)| ConvNet |
|--- |--- |
| Example: image 200x200 pixel, RGB (= 3 channels)| |
| 200x200x3 + 1 = 120.001 parameters per neuron| Kernel: 3x3 --> 3x3x3 + 1 = 28 parameters |

- Example:

    ![image2]

## Hyperparameters of Convolutional kernels <a id="conv_kernel"></a>
- There are three hyperparameters
    - Kernel size
    - Stride
    - Padding

### Kernel Size <a id="kernel_size"></a>
- Useful standard for Machine Vision: 3x3
- Also famous: 5x5, 7x7

### Stride <a id="stride"></a>
- Number of pixel steps when the kernel moves
- Standard: 1
- Sometimes: 2
- A larger stride reduces computational power, higher speed (less calculations) 

### Padding <a id="padding"></a>
- Add zeros to horizontal/vertical axis to keep the original size of the image after convolution.
- Example: 
    - Input Image 28x28, 
    - kernel 5x5, 
    - stride=1, 
    - Output Image 24x24
    - --> Output image smaller than Input image --> Add Padding to keep image dimensions 
    - --> Add two zeros to each edge

- Size of activation map:

    ![image6]

    - **D** = size of image (e.g. image size = 28x28, then D = 28)
    - **F** = size of kernel (e.g. kernel 3x3, then F = 3)
    - **P** = number of horizonal/vertical zeros (Padding)
    - **S** = Stride 

## Pooling Layers <a id="pooling_layers"></a> 
- Used in combination with Convolutional layers
- Used to reduce number of parameters and network complexity
- They speed up calculations (training)
- A pooling layer reduces space size of the activation map without changing the depth (number of kernels) in activation map.


- Alternatively: Use Global Average Layers
    - Sum up all values from activation map
    - Divide by nodes of activation map
    - One Output for each activation map

    ![image7]

### Construction <a id="pooling_construct"></a>
- Kernel, typically 2x2 
- Stride, typically 2
- --> Pooling layer evaluates at each position 4 activations but keeps only the maximum value. 
- --> It reduces the number of actiovations by the factor 4.
- Example:
    - Input: 28x28x16 (16 filter)
    - Output 14x14x16

### Aim of Convolutional and Pooling layers <a id="aim_conv_pool"></a>
- Control depth and spatial dimensions 

    ![image8]

# State of the art ConvNets <a id="state_of_art"></a>
In the following some important ConvNets will be presented.
## LeNet <a id="lenet"></a> 
- Originally from 1998
- Here LeNet with some modern implementations due to more computational power in these days
- Open Jupyter Notebook ```lenet_in_keras.ipynb```
    ### Load dependencies
    ```
    import keras
    from keras.datasets import mnist
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.layers import Flatten, Conv2D, MaxPooling2D # new!
    ```
    ### Load data
    ```
    (X_train, y_train), (X_valid, y_valid) = mnist.load_data()
    ```
    ### Preprocess Data
    ```
    X_train = X_train.reshape(60000, 28, 28, 1).astype('float32')
    X_valid = X_valid.reshape(10000, 28, 28, 1).astype('float32')

    X_train /= 255
    X_valid /= 255

    n_classes = 10
    y_train = keras.utils.to_categorical(y_train, n_classes)
    y_valid = keras.utils.to_categorical(y_valid, n_classes)
    ```
    ### Design neural network architecture
    ```
    model = Sequential()

    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))

    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())

    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(n_classes, activation='softmax'))
    ```
    ```
    model.summary()

    RESULTS:
    ------------
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 26, 26, 32)        320       
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 24, 24, 64)        18496     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 12, 12, 64)        0         
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 12, 12, 64)        0         
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 9216)              0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 128)               1179776   
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 128)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 10)                1290      
    =================================================================
    Total params: 1,199,882
    Trainable params: 1,199,882
    Non-trainable params: 0
    ```
    ### Configure model
    ```
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    ```
    ### Train
    ```
    model.fit(X_train, y_train, batch_size=128, epochs=10, verbose=1, validation_data=(X_valid, y_valid))

    RESULTS:
    ------------
    ...
    Epoch 10/10
    60000/60000 [==============================] - 39s 655us/step - loss: 0.0264 - acc: 0.9914 - val_loss: 0.0291 - val_acc: 0.9919
    ```
    **Structure**:
    - Input: MNIST 28x28
    - Two Conv layers (1st: 32 filter, to learn simple features, 2nd: 64 filter, to learn more complex feature combinations)
    - kernel_size: 3x3
    - ReLu as activation
    - Stride: 1
    - Padding: valid (--> means: no Padding, shrinkage of original image)
    - Output of 1st Conv layer: 26x26x32
    - Output of 1st Conv layer: 24x24x64
    - MaxPooling2D to reduce spatial dimensions (and hence computational power), kernel: 2x2, Stride: 2
    - Output of MaxPooling2D: 12x12x64 
    - Dropout to reduce risk of overfitting
    - Flatten --> Convert 3D activation map into a 1D array, 9216 neuron (12x12x64)
    - Two Dense layers to prepare output, 1st dense layer with 128 neurons, 2nd dense layer with 10 neurons, used to learn to assign representations to classes
    - Softmax-output at the end

    **Parameters**:
    - conv2d_1: 320 params
        - 288 weights: 32 filter x 9 weights (from 3x3 kernel * 1 channel) 
        - 32 Bias, one for each filter
    - conv2d_2: 18.496 params
        - 18.432 weights: 64 filter x 9 weights per filter (32 in total from previous layer) 
        - 64 Bias, one for each filter
    - dense_1: 1.179.648 params
        - 9216 inputs from reduced activation map of the previous layer x 128 neurons of this layer
        - 128 Bias, one for each neuron
    - dense_2: 1290 params
        - 128 inputs from reduced activation map of the previous layer x 10 neurons of this layer
        - 10 Bias, one for each neuron
    
    Prinicipal structure based on LeNet:

    ![image9]

## AlexNet <a id="alexnet"></a> 
- Open Jupyter Notebook ```alexnet_in_keras.ipynb```
    ### Load dependencies
    ```
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
    from keras.layers.normalization import BatchNormalization
    ```
    ### Load data
    ```
    import tflearn.datasets.oxflower17 as oxflower17
    X, Y = oxflower17.load_data(one_hot=True)
    ```
    ### Design neural network architecture
    ```
    model = Sequential()

    model.add(Conv2D(96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(256, kernel_size=(5, 5), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    model.add(Conv2D(256, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(384, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(384, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(4096, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='tanh'))
    model.add(Dropout(0.5))

    model.add(Dense(17, activation='softmax'))
    ```
    ```
    model.summary()

    RESULTS:
    ------------
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 54, 54, 96)        34944     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 26, 26, 96)        0         
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 26, 26, 96)        384       
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 22, 22, 256)       614656    
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 10, 10, 256)       0         
    _________________________________________________________________
    batch_normalization_2 (Batch (None, 10, 10, 256)       1024      
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 8, 8, 256)         590080    
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 6, 6, 384)         885120    
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 4, 4, 384)         1327488   
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 1, 1, 384)         0         
    _________________________________________________________________
    batch_normalization_3 (Batch (None, 1, 1, 384)         1536      
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 384)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 4096)              1576960   
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 4096)              0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 4096)              16781312  
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 4096)              0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 17)                69649     
    =================================================================
    Total params: 21,883,153
    Trainable params: 21,881,681
    Non-trainable params: 1,472
    ```
    ### Configure model
    ```
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    ```
    ### Train
    ```
    model.fit(X, Y, batch_size=64, epochs=100, verbose=1, validation_split=0.1, shuffle=True)

    RESULTS:
    ------------
    Train on 1224 samples, validate on 136 samples
    Epoch 1/100
    1224/1224 [==============================] - 21s 17ms/step - loss: 4.5644 - acc: 0.2092 - val_loss: 7.3530 - val_acc: 0.1691
    ...
    ```
    Key points:
    - Input images: 224x224, RGB --> 3 color channels considered via input_shape of conv2d_1
    - conv2d_1 has a large kernel_size = 11x11
    - Dropout is only used in for the fully connected layers near the output. Idea: Conv layers are less susceptible for overfitting than fully connected layers.

    ![image10]

## VGGNet <a id="vggnet"></a> 
- VGGNet is similar to AlexNet 
- Main difference: VGGNet has **more Conv-Pool-Blocks** than AlexNet
- Open Jupyter Notebook ```vggnet_in_keras.ipynb```
    ### Load and prepare 
    ```
    ... (see notebook)
    ```
    ### Design neural network architecture
    ```
    model = Sequential()

    model.add(Conv2D(64, 3, activation='relu', input_shape=(224, 224, 3)))
    model.add(Conv2D(64, 3, activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(BatchNormalization())

    model.add(Conv2D(128, 3, activation='relu'))
    model.add(Conv2D(128, 3, activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(BatchNormalization())

    model.add(Conv2D(256, 3, activation='relu'))
    model.add(Conv2D(256, 3, activation='relu'))
    model.add(Conv2D(256, 3, activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(BatchNormalization())

    model.add(Conv2D(512, 3, activation='relu'))
    model.add(Conv2D(512, 3, activation='relu'))
    model.add(Conv2D(512, 3, activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(BatchNormalization())

    model.add(Conv2D(512, 3, activation='relu'))
    model.add(Conv2D(512, 3, activation='relu'))
    model.add(Conv2D(512, 3, activation='relu'))
    model.add(MaxPooling2D(2, 2))
    model.add(BatchNormalization())

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))

    model.add(Dense(17, activation='softmax'))
    ```
    ```
    model.summary() 

    RESULTS:
    ------------
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_1 (Conv2D)            (None, 222, 222, 64)      1792      
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 220, 220, 64)      36928     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 110, 110, 64)      0         
    _________________________________________________________________
    batch_normalization_1 (Batch (None, 110, 110, 64)      256       
    _________________________________________________________________
    conv2d_3 (Conv2D)            (None, 108, 108, 128)     73856     
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 106, 106, 128)     147584    
    _________________________________________________________________
    max_pooling2d_2 (MaxPooling2 (None, 53, 53, 128)       0         
    _________________________________________________________________
    batch_normalization_2 (Batch (None, 53, 53, 128)       512       
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 51, 51, 256)       295168    
    _________________________________________________________________
    conv2d_6 (Conv2D)            (None, 49, 49, 256)       590080    
    _________________________________________________________________
    conv2d_7 (Conv2D)            (None, 47, 47, 256)       590080    
    _________________________________________________________________
    max_pooling2d_3 (MaxPooling2 (None, 23, 23, 256)       0         
    _________________________________________________________________
    batch_normalization_3 (Batch (None, 23, 23, 256)       1024      
    _________________________________________________________________
    conv2d_8 (Conv2D)            (None, 21, 21, 512)       1180160   
    _________________________________________________________________
    conv2d_9 (Conv2D)            (None, 19, 19, 512)       2359808   
    _________________________________________________________________
    conv2d_10 (Conv2D)           (None, 17, 17, 512)       2359808   
    _________________________________________________________________
    max_pooling2d_4 (MaxPooling2 (None, 8, 8, 512)         0         
    _________________________________________________________________
    batch_normalization_4 (Batch (None, 8, 8, 512)         2048      
    _________________________________________________________________
    conv2d_11 (Conv2D)           (None, 6, 6, 512)         2359808   
    _________________________________________________________________
    conv2d_12 (Conv2D)           (None, 4, 4, 512)         2359808   
    _________________________________________________________________
    conv2d_13 (Conv2D)           (None, 2, 2, 512)         2359808   
    _________________________________________________________________
    max_pooling2d_5 (MaxPooling2 (None, 1, 1, 512)         0         
    _________________________________________________________________
    batch_normalization_5 (Batch (None, 1, 1, 512)         2048      
    _________________________________________________________________
    flatten_1 (Flatten)          (None, 512)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 4096)              2101248   
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 4096)              0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 4096)              16781312  
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 4096)              0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 17)                69649     
    =================================================================
    Total params: 33,672,785
    Trainable params: 33,669,841
    Non-trainable params: 2,944
    ```

    ![image11]

## ResNet <a id="resnet"></a> 
- ResNet architectures help to reduce the risk of **Vanishing Gradients**
- Vanishing Gradients are remarkable for deep architectures
- Reason: Parameter of earl layers are far away from cost function (source for gradients which will transported backward through the network)
- Early layers (needed for simple feature detection) are difficult to train.

### Identity functions
- If new layers execute an identity check (reproduction of results from earlier layers) training error does not increase.
- However, often new layers have problems to execute this check. Hence, new layers could reduce the performance.

### Idea of ResNets:
- Residual networks have residual connections which exist in so called **residual blocks**.
- Residual blocks are based on a sequence like Convolution layers, Batchnormalization and ReLU activation which will be finalized by a residual connection.
- Input to residual block: a<sub>i-1</sub>.
- Output from residual block (w/o residual connection): a<sub>i</sub>.
- Output from residual block (with residual connection): y<sub>i</sub> = a<sub>i</sub> + a<sub>i-1</sub>.
- **If activation from residual block a<sub>i</sub> = 0 (no learning effect), then the final output is the original input,  y<sub>i</sub> = a<sub>i-1</sub>**
- **Hence, in that way a residual block is an identity function. A residual block learns something useful and rdcuces the error or it does noting (identity function)**. Residual connections are also called **skip connections**. --> **Neutral or better** property.
- The risk of Vanishing Gradient problems can be reduced in that way.

    ![image12]

    ![image13]


## Object detection <a id="object_detection"></a> 
### R-CNN <a id="r_cnn"></a> 
### Fast R-CNN <a id="fast_r_cnn"></a> 
### Faster R-CNN <a id="faster_r_cnn"></a> 
### YOLO <a id="yolo"></a> 
## Picture Segmentation <a id="pic_seg"></a> 
### Mask R-CNN <a id="mask_r_cnn"></a> 
### U-Net <a id="u_net"></a> 
## Transfer Learning <a id="trans_learn"></a> 
## Capsule Networks <a id="caps_net"></a>  

## Setup Instructions <a id="Setup_Instructions"></a>
The following is a brief set of instructions on setting up a cloned repository.

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites: Installation of Python via Anaconda and Command Line Interaface <a id="Prerequisites"></a>
- Install [Anaconda](https://www.anaconda.com/distribution/). Install Python 3.7 - 64 Bit

- Upgrade Anaconda via
```
$ conda upgrade conda
$ conda upgrade --all
```

- Optional: In case of trouble add Anaconda to your system path. Write in your CLI
```
$ export PATH="/path/to/anaconda/bin:$PATH"
```

### Clone the project <a id="Clone_the_project"></a>
- Open your Command Line Interface
- Change Directory to your project older, e.g. `cd my_github_projects`
- Clone the Github Project inside this folder with Git Bash (Terminal) via:
```
$ git clone https://github.com/ddhartma/Matrix-Math-with-Numpy.git
```

- Change Directory
```
$ cd Matrix-Math-with-Numpy
```

- Create a new Python environment, e.g. matrix_op. Inside Git Bash (Terminal) write:
```
$ conda create --id matrix_op
```

- Activate the installed environment via
```
$ conda activate matrix_op
```

- Install the following packages (via pip or conda)
```
numpy = 1.12.1
pandas = 0.23.3
matplotlib = 2.1.0
seaborn = 0.8.1
```

- Check the environment installation via
```
$ conda env list
```

## Acknowledgments <a id="Acknowledgments"></a>
* This project is part of the Udacity Nanodegree program 'Deep Reinforcement Learning'. Please check this [link](https://www.udacity.com) for more information.

## Further Links <a id="Further_Links"></a>

Git/Github
* [GitFlow](https://datasift.github.io/gitflow/IntroducingGitFlow.html)
* [A successful Git branching model](https://nvie.com/posts/a-successful-git-branching-model/)
* [5 types of Git workflows](https://buddy.works/blog/5-types-of-git-workflows)

Docstrings, DRY, PEP8
* [Python Docstrings](https://www.geeksforgeeks.org/python-docstrings/)
* [DRY](https://www.youtube.com/watch?v=IGH4-ZhfVDk)
* [PEP 8 -- Style Guide for Python Code](https://www.python.org/dev/peps/pep-0008/)

Important web sites - Deep Learning
* Deep Learning - illustriert - [GitHub Repo](https://github.com/the-deep-learners/deep-learning-illustrated)
* Jason Yosinski - [Visualize what kernels are doing](https://www.youtube.com/watch?v=AgkfIQ4IGaM)

Further Resources
* Read about the [WaveNet](https://deepmind.com/blog/article/wavenet-generative-model-raw-audio) model. Why train an A.I. to talk, when you can train it to sing ;)? In April 2017, researchers used a variant of the WaveNet model to generate songs. The original paper and demo can be found [here](https://arxiv.org/pdf/1609.03499.pdf).
* Learn about CNNs [for text classification](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/). You might like to sign up for the author's [Deep Learning Newsletter!](https://www.getrevue.co/profile/wildml)
* Read about Facebook's novel [CNN approach for language translation](https://engineering.fb.com/2017/05/09/ml-applications/a-novel-approach-to-neural-machine-translation/) that achieves state-of-the-art accuracy at nine times the speed of RNN models.
* Play [Atari games with a CNN and reinforcement learning](https://deepmind.com/research/publications/human-level-control-through-deep-reinforcement-learning). If you would like to play around with some beginner code (for deep reinforcement learning), you're encouraged to check out Andrej Karpathy's [post](http://karpathy.github.io/2016/05/31/rl/).
* Play [pictionary](https://quickdraw.withgoogle.com/#) with a CNN! Also check out all of the other cool implementations on the [A.I. Experiments](https://experiments.withgoogle.com/collection/ai) website. Be sure not to miss [AutoDraw](https://www.autodraw.com/)!
* Read more about [AlphaGo]. Check out [this article](https://www.technologyreview.com/2017/04/28/106009/finding-solace-in-defeat-by-artificial-intelligence/), which asks the question: If mastering Go “requires human intuition,” what is it like to have a piece of one’s humanity challenged?
* Check out these really cool videos with drones that are powered by CNNs.
    - Here's an interview with a startup - [Intelligent Flying Machines (IFM)](https://www.youtube.com/watch?v=AMDiR61f86Y).
    - Outdoor autonomous navigation is typically accomplished through the use of the [global positioning system (GPS)](www.droneomega.com/gps-drone-navigation-works/), but here's a demo with a CNN-powered [autonomous drone](https://www.youtube.com/watch?v=wSFYOw4VIYY).

* If you're excited about using CNNs in self-driving cars, you're encouraged to check out:
    - Udacity [Self-Driving Car Engineer Nanodegree](https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013), where we classify signs in the German Traffic Sign dataset in this project.
    - Udacity [Machine Learning Engineer Nanodegree](https://www.udacity.com/course/machine-learning-engineer-nanodegree--nd009t), where we classify house numbers from the Street View House Numbers dataset in this project.
    - This series of blog posts that details how to train a CNN in Python to produce a self-driving A.I. to play Grand Theft Auto V.

* Check out some additional applications not mentioned in the video.
    - Some of the world's most famous paintings have been [turned into 3D](https://www.businessinsider.com/3d-printed-works-of-art-for-the-blind-2016-1) for the visually impaired. Although the article does not mention how this was done, we note that it is possible to use a CNN to [predict depth](https://cs.nyu.edu/~deigen/depth/) from a single image.
    - Check out [this research](https://ai.googleblog.com/2017/03/assisting-pathologists-in-detecting.html) that uses CNNs to localize breast cancer.
    - CNNs are used to [save endangered species](https://blogs.nvidia.com/blog/2016/11/04/saving-endangered-species/?adbsc=social_20170303_70517416)!
    - An app called [FaceApp](https://www.digitaltrends.com/photography/faceapp-neural-net-image-editing/) uses a CNN to make you smile in a picture or change genders.

