# veritas

## About

Veritas is a sandbox playground for developing, integrating, and testing robotic platforms.

It uses [Gensis](https://github.com/Genesis-Embodied-AI/Genesis) as the backend for the simulation environment.

Also utilizing WandB for experiment tracking. It does require an account but it is free

## Setup

1. Clone the repository. Be sure to pull git lfs objects: `git lfs pull`

2. Install Genesis and its dependencies. Ensure torch is installed properly. Refer to the [Genesis installation guide](https://github.com/Genesis-Embodied-AI/Genesis?tab=readme-ov-file#quick-installation)

3. Install W&B: `pip install wandb`  
   Log into W&B: `wandb login` and follow the prompts to authenticate your account.

## Veritas Flow

- Load robot model into Genesis using URDF file
- Define IK workspace for a linkage utilizing the `scene_test.py` file as a reference. This file has the tools to load a simple tkinter UI to help define and save the workspace. This will save off the workspace to `ik_workspace.json`.
- Generate the sampling points by calling `dataset.py`. This will create a pickle file with the sampled points and corresponding joint angles from the IK solver. This will be saved off as `ik_dataset.pkl`.
- Call the training script `train.py`. This will load the dataset in and train the IK net model using supervised learning.
- Once trained, the model will be saved off in the checkpoints folder. There will be both a `ik_model.pt` file that contains the model weights, and a `buffer.pkl`, which contains the sampled points used for training (helpful for understanding if there were gaps in the training data using onion plot).
- You can run `evaluate_model.py` and/or `onion_plot.py` to evaluate the model and visualize the results.

## Sampling Points

Points for sampling are generated using the `generate_buffer` function in `utils.py`. This samples points within a spherical shell and outside a cylinder, maintaining a range inside of the physical reach of the robot arm. The range sampling is done to ensure the IK solver has an optimal solution for any point so the mean time calculation for the IK solver is not skewed by points that are out of reach where the solver has to iterate util it hits the max iteration limit. This is also done exported as a static list of points in a tensor to ensure high speed with parallelization during training. If you are trying to reuse this sampling method and they are grouped too closely together, simply increase the step size until it is evenly distributed. Here is an example of the points sampled using the onion plot utility in `onion_plot.py`:

![Example Sampled Points](docs/example_sampling.png)

## Reward Function

The reward is calculated based on the difference of angles between the IK model joints and the NN model joints output, with a bonus based on the distance to the target point.

# IK Solver

The default solver for genesis is being utilized. It obtains roughly a mean error of 0.0110m and a standard deviation of 0.0187m.

## Approach and Results

### Training attempt 1

First training method used was simply getting the model's joint angles, and would sample a random point from a gaussian distribution using the model's joint output as the mean. It would then propagate the loss through the network based on the reward function to improve the model. The results are as follows:

| Steps   | Mean Error (m) | Standard Deviation (m) |
| ------- | -------------- | ---------------------- |
| 1,000   | 0.8808         | 0.2668                 |
| 10,000  | 0.7251         | 0.2344                 |
| 100,000 | 0.5594         | 0.2394                 |

While this shows some improvement with increased steps, the model still has a high error rate and a quite shallow learning curve for the number of training steps. I am only testing with my physical hardware at home, so focusing on optimizing the training loop should be the next step.

### Training attempt 2

The next approach for training is utilizing a modified Curriculum learning approach. I believe this will help the model learn more effectively as it will be able to learn each point before moving onto the next. The goal is to train it on a specific buffer of points, that starts at size 1, and increments once the model can reach a certain performance threshold on the current buffer. The first run was allowed to save off the model every time it completed 10,000 steps, but this was later changed to only save when the buffer size was increased. This led to a model that sovled 9 seprate points in less than 400k steps, but couldn't solve 10 in over 4M steps. The second and third run do not make it to 10 values in the buffer, but have some crazy diverging occuring with the loss function going all the way past 10,000 in some cases. The results are as follows:

| Steps     | Total Buffer Size | Mean Error (m) | Standard Deviation (m) |
| --------- | ----------------- | -------------- | ---------------------- |
| 4,000,000 | 10                | 0.5113         | 0.2859                 |
| 1,000,000 | 8                 | 0.4689         | 0.2910                 |
| 300,000   | 7                 | 0.4971         | 0.2086                 |

With the base curriculum learning approach, the model is able to get to a lower mean error, however suffered from a high standard deviation in most cases. The loss diverging instead of converging is a concern, but this method has brought more unknowns then answers for the next step.

### Training attempt 3

This appoach is very similar to v2, but with a more gradual approach to the curriculum learning. Instead of sampling a random point anywhere, it samples a point a random disatnce away from the previous target point within a small threshold. This is to help the model learn with less extreme joint movements until it can sample more points. The second thing that is added is an increasing in the reward threshold as the buffer size increases. This is to help the model progress through the curriculum and learn a more generalized solution. Lastly logging for the list of active points in the buffer is added to help with post processing and analysis, along with a new onion like plot aiming to find convergence zones where the model has been specifically trained on. This method has three main issues - 1: there are cases where the system will diverge and the loss will go through the roof, taking nearly half a million steps to recover. 2: This approach is not scalable. As the model approaches a buffer of 1000 elements, it quickly becomes clear that running it across all 1000 elements takes a significant amount of time. This can be easily visualized in the plot below as the steps run with consistent timing, where cycle is a function of running through the buffer. This indicates an exponential impact as the buffer is increased to better train the model. 3: The model seems to hit a wall at around 0.5m mean error and struggles to get lower.

| Steps     | Total Buffer Size | Mean Error (m) | Standard Deviation (m) |
| --------- | ----------------- | -------------- | ---------------------- |
| 1,200,000 | 1,000             | 0.5029         | 0.2077                 |

![Cycle Vs Steps Plot](docs/cycle_vs_steps.png)


### Training Attempt 4

I spent more time optimizing this development cycle, and found a few bugs in the core loop that was causing errors and instability. Once removing all of that, I focused on cleaning up the training loop, removing curriculum learning, and variable hyperparameter optimization based on the training accuracy to help get a more stable training process. This led to a much repeatable training process that could reach lower mean error values then seen in the other iterations in only a fraction of the steps. Once this was done, I focused on hyperparameter optimization using WandB sweeps to find the best parameters for training and will start doing that for all future training attempts. This will allow for a more systematic approach to compare training methods and not rely on arbitrary hyperparameter selection. The results of the hyperparameter sweep are below, showing a clear optimal set of parameters, mostly based on reducing both the learning rate and cation standard deviation to around 0.02. Sweep is shown below:

![Hyperparameter Sweep](docs/hyperparameter_sweep.png)  

It should be noted that I made the move over to a lower dimensional robot arm for this training, closer to what I would be expecting for a legged robot. It limits it to 3 DOF, which may be where some of the increased accuracy and faster convergence during training is coming from. The results of the best model from the sweep are below:

| Steps     | Total Buffer Size | Mean Error (m) | Standard Deviation (m) |
| --------- | ----------------- | -------------- | ---------------------- |
| 5,000     | 1,000             | 0.0153         | 0.0073                 |

As seen here, this is a large leap in accuracy compared to previous attempts with over a 30x reduction in mean error and just over a 27x reduction in the standard devaition of that error. There is still work to be done here for improving accuracy before this can be used in a live system, looking over the onion plot (2D representation of the 3D workspace) there are still some holes in the training where the model is not converging well. It isn't for the lack of sampling at those points, the optimizer is just choosing to not converge there. Future work will be focused on lowering the overall error more, removing the non-converged holes, and aimed at approaching the accuracy of the default IK solver. It should be noted the IK solver for the 3DOF is significanly faster at only 3.34ms when run on the gpu, but even still the neural network is achieving 0.22ms, a 15x speed up. The onion plot is shown below:

![Onion Plot](docs/onion_plot_training_attempt_4.png)