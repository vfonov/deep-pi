# DEEP PI

## An example of running deep neural-net image classifier on Raspberry PI

### Installing torch on Raspbian
If you are using minimal version of Raspbian, you will need to install several packages first:
```
sudo apt-get install -y build-essential gcc g++ curl  cmake libreadline-dev libjpeg-dev libpng-dev ncurses-dev imagemagick gfortran libopenblas-base libopenblas-dev
```
The clone [torch](http://torch.ch/) distribution:
```
git clone https://github.com/torch/distro.git ~/torch --recursive
```
And start building (takes several hours on Raspberry PI B+:
```
cd ~/torch
./install.sh
```
If you encounter following error: 
```...In function ‘THByteVector_vectorDispatchInit’: /home/pi/torch/pkg/torch/lib/TH/generic/simd/simd.h:64:3: error: impossible constraint in ‘asm’ ...```  
it means that you are building on a cpu without NEON extension (the kind Raspberry PI Version A & B have). You will need to checkout latest version of torch and disable submodule update command in install.sh script ( comment out line 45 in ~/torch/install.sh ) and then update torch torch:
```
cd ~/torch/pkg/torch/
git checkout master
git pull
```
and run `./install.sh` script again. 

After ./install.sh is finished - it will ask if you want to update .bashrc to include call to initialize torch environment every time you login. If you  don't want it, you will have to execute command `. ~/torch/install/bin/torch-activate` before you will be able to lauch th. 

### Alternative way to install torch on Raspbian, using precompiled blob
I created an archive of torch installation compiled for Raspberry PI B+ , running Raspbian 8 
You can download it here : https://github.com/vfonov/deep-pi/releases/download/v1/torch_intstall_raspbian_arm6l_20161218.tar.gz 
Copy file to /home/pi, then run `tar zxf torch_intstall_raspbian_arm6l_20161218.tar.gz` - this will create torch subdirectory that will include only precompiled binaries. To activate it add `. torch/install/bin/torch-activate` in the end of the `~/.bashrc` file. 

### Running MNIST digit classifier from torch demos
You can install various torch example from https://github.com/torch/demos, here is an output from MNIST digit classieifer training session:

```
pi:~/src/demos/train-a-digit-classifier $ th train-on-mnist.lua 
<torch> set nb of threads to 4	
<mnist> using model:	
nn.Sequential {
  [input -> (1) -> (2) -> (3) -> (4) -> (5) -> (6) -> (7) -> (8) -> (9) -> (10) -> output]
  (1): nn.SpatialConvolutionMM(1 -> 32, 5x5)
  (2): nn.Tanh
  (3): nn.SpatialMaxPooling(3x3, 3,3, 1,1)
  (4): nn.SpatialConvolutionMM(32 -> 64, 5x5)
  (5): nn.Tanh
  (6): nn.SpatialMaxPooling(2x2, 2,2)
  (7): nn.Reshape(576)
  (8): nn.Linear(576 -> 200)
  (9): nn.Tanh
  (10): nn.Linear(200 -> 10)
}
<warning> only using 2000 samples to train quickly (use flag -full to use 60000 samples)	
<mnist> loading only 2000 examples	
<mnist> done	
<mnist> loading only 1000 examples	
<mnist> done	
<trainer> on training set:	
<trainer> online epoch # 1 [batchSize = 10]	
 [===================>.................... 471/2000 ....................................]  ETA: 2m20s | Step: 92ms      
```
Overall it is about 5 times slower then running the same example on a desktop with Core i5 @  3.30GHz without using GPU. 

### Installing deep-pi
```
git clone https://github.com/vfonov/deep-pi 
```
After that you can launch `download_net.sh` script to download the pretrained NIN network ( based on https://gist.github.com/szagoruyko/0f5b4c5e2d2b18472854 ) to the `/home/pi` path. **WARNING** pretrained network is 33Mb file!


#### Running 
To run on a single image: `th test_single.lua <path to your image>` 
To run continious classification using frames from camera ( I recommend using external USB camera) :
```
nohup th -ldisplay.start 8000 0.0.0.0 & 
th camera_interface.lua
```
Then open web browser and point to to location http://your.raspberry.ip:8000  - replace your.raspberry.ip with IP address that your Raspberry PI is configured to use. 

#### Setup 
![Camera and test object](https://cloud.githubusercontent.com/assets/628822/21299836/637e738a-c56d-11e6-80a4-c20605527d89.jpg)

#### Output 
![Example 1](https://cloud.githubusercontent.com/assets/628822/21299835/637e6700-c56d-11e6-9c01-8e600417ac4d.jpg)

![Example 2](https://cloud.githubusercontent.com/assets/628822/21299834/637e11ce-c56d-11e6-82e1-c78ebf69004b.jpg)

![Example 3](https://cloud.githubusercontent.com/assets/628822/21299833/637df9b4-c56d-11e6-8f06-6c4e22f45957.jpg)

