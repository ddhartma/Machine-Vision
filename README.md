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

- In case of RGB: Each filter exists for each color channel

    ![image4]
    
    - In order to calculate the output you have to take the RELU triggered weighted sum over all color channels as shown above. 

- In case of multiple filter (here 3):

    ![image5]

    In case of early Conv-layers:
    - detection of simple features like edges. Those filters operate like markers, each for a certain feature
    - One filter for horizontal edges
    - One filter for vertical edges 
    - One filter for color transitions
    - etc. 

    In case of deeper Conv-layers:
    - complexer combinations of simple features (like textures and certain shapes, forms)

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
## LeNet <a id="lenet"></a> 
## AlexNet <a id="alexnet"></a> 
## VGGNet <a id="vggnet"></a> 
## ResNet <a id="resnet"></a> 
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

