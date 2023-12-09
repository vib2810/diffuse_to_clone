# diffuse_to_clone
Diffusion Models for Imitation Learning
## Pre-requisites:
1. Install Docker
2. Install Nvidia Docker
3. Work with a system that has the franka arm configured with the [frankapy](https://github.com/iamlab-cmu/frankapy) package

## Setup:
1. Make Docker Container (use sudo if docker is not added to the user group):
   ```
   docker build -t diffusion .
   ```
   Use `newgrp docker` if you didnt logout 

## Running the code:
0. Unlock robot joints
    - `ssh -X student@[control-pc-name]`
    - `google-chrome`
    - In firefox open `https://172.16.0.2/desk/` and press `Click to unlock joints`
      
1. Run the start_control_pc script from the frankapy package
   - `cd <frankapy package directory>`
   - `bash ./bash_scripts/start_control_pc.sh -u student -i [control-pc-name]`

2. Run the MoveIt Server
    - Start the docker container
      ```
      bash run_docker.sh
      ```
    - Inside the container:
      ```
      roslaunch manipulation demo_moveit.launch 
      ```
3. Collect data with vision:
      ```
      roslaunch manipulation data_collect_vision.launch
      ```

      Launch realsense from inside PC
      ```
      export ROS_IP=172.26.230.217 && export ROS_MASTER_URI=http://172.26.165.201:11311/ && roslaunch realsense2_camera rs_camera.launch 
      ```
4. SCP command:
      - Transfer Dataset
         - All with scp:
            scp -r /home/vib2810/diffuse_to_clone/dataset/data/vision_new_camera guest@punisher.wifi.local.cmu.edu:/home/guest/vibhakar/diffuse_to_clone/dataset
         - Transfer only new files (requires to run bash claim_files.sh on punisher)
            rsync -avz --ignore-existing /home/vib2810/diffuse_to_clone/dataset/data/vision_new_camera guest@punisher.wifi.local.cmu.edu:/home/guest/vibhakar/diffuse_to_clone/dataset/data/

      - Transfer Models from Punisher to Local
            - scp -r guest@punisher.wifi.local.cmu.edu:/home/guest/vibhakar/diffuse_to_clone/logs/models/* /home/vib2810/diffuse_to_clone/logs/models/



