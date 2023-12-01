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
