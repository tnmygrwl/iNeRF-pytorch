This repository has been built over this pytorch implementation of NeRF: https://github.com/yenchenlin/nerf-pytorch

CONTRIBUTION:
    Tanmay - Implemented inerf_sampling.py
    Videsh - Adapted and implemented so3_helpers.py
    Both - pose_estimation function in run_inerf.py, results.py   

DEPENDENCIES:
    - pytorch
    - torchvision
    - numpy
    - imageio
    - imageio-ffmpeg
    - matplotlib
    - configargparse
    - opencv-python
    - pyquaternion

INSTALLATION:
    Run the following commands.
        cd inerf
        pip install -r requirements.txt


IMPLEMENTATION DETAILS:

    Currently, this implementation supports 2 datasets i.e. nerf-synthetic and nerf-llff. Performing pose estimation
    for any query image of a scene also requires its pretrained NeRF representation. This repository may not contain 
    all the details to train NeRF. We have evaluated results for scenes lego (synthetic) and fern (llff). 
    
    run_inerf.py script performs pose estimation for a random query image from the validation/test dataset.
    The pose is randomly initialized in some vicinity of the ground truth camera pose.
    Please check out the report for more details.

    It usually takes less than 5 minutes to optimize pose for 1 query image on a single GPU.

    Data: 
        - lego (synthetic): pretrained NeRF, images and corresponding camera poses
        - fern (llff): pretrained NeRF, images and corresponding camera poses

    How to optimize camera pose:
        python run_inerf.py --config inerf_configs/{SCENE}.txt    (replace {SCENE} with lego | fern)
    
    
    Also, for more data, pretrained models and other implementation details on NeRF, checkout the file nerf_README.md


CODE CITATIONS:
    - Pytorch NeRF implementation: https://github.com/yenchenlin/nerf-pytorch
    - Modern Robotics for exponential parameterization: https://github.com/NxRLab/ModernRobotics/blob/master/packages/Python/modern_robotics/core.py
