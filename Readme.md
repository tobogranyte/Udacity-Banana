# Project Environment

The project environment is the Banana Unity ML environment specifically created for the Udacity course as part of the [DRLND repository](https://github.com/udacity/deep-reinforcement-learning#dependencies). 

## State Space

The state space has 37 dimensions.

## Action Space

The action space has 4 dimensions corresponding to forward, backward, turn left, and turn right.


## Solving the Environment

The environment is a square containing blue and yellow bananas which, as the agent navigates the environment, get picked up when the agent navigates over a banana (blue or yellow). There is a reward of -1 for picking up a blue banana and +1 for picking up a yellow banana. The environment is considered solved when the average reward over 100 episodes is greater than 13.

# The Project Code

## Python

To run, the project requires a correctly configured Python environment. Follow the instructions for dependencies [here](https://github.com/udacity/deep-reinforcement-learning#dependencies) to do this.

You will also need to install `unityagents`. You can do this with the following command:

    pip install unityagents

## Environments

The project is configured to run on either Linux or Mac. In the root folder of the project are the two supported Banana environments:

* Banana.app
* banana_linux (folder)

If each of these are present in the root folder (as they are when the project is clones), the project should run correctly for Linux and Mac.

# Running the Code

## From Jupyter Notebook

* From the `Banana Project` directory, open the Jupyter notebook `Banana_Project.ipynb`.
* Make sure you are using a kernel in the drlnd environment you created [here](https://github.com/udacity/deep-reinforcement-learning#dependencies).
* If a different kernel is active, from the menu, select Kernel > Change kernel > drlnd.
* Run each of the cells in **Setup Blocks**
* To train the agent, run the final block in **Train the Agent**

## From Command Line

* From the `Banana Project` directory, enter the following:

    python Banana_Project.py
