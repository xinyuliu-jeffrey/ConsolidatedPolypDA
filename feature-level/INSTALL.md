## Installation

### Requirements:
- PyTorch 1.0 from a nightly release. Installation instructions can be found in https://pytorch.org/get-started/locally/
- torchvision from master
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- (optional) OpenCV for the webcam demo


### Step-by-step installation

```bash
# first, make sure that your conda is setup properly with the right environment
# for that, check that `which conda`, `which pip` and `which python` points to the
# right path. From a clean conda env, this is what you need to do

conda create --name ConsolidatedPolypDA
source activate ConsolidatedPolypDA

# this installs the right pip and dependencies for the fresh python
conda install ipython

# ConsolidatedPolypDA and coco api dependencies
pip install ninja yacs cython matplotlib

# follow PyTorch installation in https://pytorch.org/get-started/locally/
# we give the instructions for CUDA 9.0
conda install pytorch-nightly -c pytorch

# install torchvision
cd ~/github
git clone https://github.com/pytorch/vision.git
cd vision
python setup.py install

# install pycocotools
cd ~/github
git clone https://github.com/cocodataset/cocoapi.git
cd cocoapi/PythonAPI
python setup.py build_ext install

# install PyTorch Detection
cd ~/github
git clone https://github.com/xinyuliu-jeffrey/ConsolidatedPolypDA.git
cd ConsolidatedPolypDA
cd feature-level
# the following will install the lib with
# symbolic links, so that you can modify
# the files if you want and won't need to
# re-build it
python setup.py build develop

# or if you are on macOS
# MACOSX_DEPLOYMENT_TARGET=10.9 CC=clang CXX=clang++ python setup.py build develop
```