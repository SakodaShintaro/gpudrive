#!/usr/bin/env python
# coding: utf-8

# ## Scenario loading and structure
#
# `GPUDrive` is a multi-agent driving simulator built on top of the [Waymo Open Motion Dataset (WOMD)](https://waymo.com/open/) (See also [Ettinger et al., 2021](https://arxiv.org/abs/2104.10133)).
#
# In this tutorial, we explain the structure of a traffic scenario and show use processed scenario data with `GPUDrive`.
#
# **Useful links to learn more**:
# - [`waymo-open-dataset`](https://github.com/waymo-research/waymo-open-dataset): Official dataset repo
# - [tf.Example proto format](https://waymo.com/open/data/motion/tfexample): Data dictionary for a raw WOMD scenario
# - [GPUDrive `data_utils`](https://github.com/Emerge-Lab/gpudrive/tree/main/data_utils): Docs and code we use to process the WOMD scenarios


import json
import matplotlib.pyplot as plt
import os
from pathlib import Path
import pandas as pd
from gpudrive.env.dataset import SceneDataLoader
from gpudrive.env.env_torch import GPUDriveTorchEnv
from gpudrive.env.config import EnvConfig


# Set working directory to the base directory
working_dir = Path(__file__).resolve().parents[2]
print(f"{working_dir=}")
os.chdir(working_dir)

data_loader = SceneDataLoader(
    root="data/processed/examples",  # Path to the dataset
    batch_size=10,  # Batch size, you want this to be equal to the number of worlds (envs) so that every world receives a different scene
    dataset_size=4,  # Total number of different scenes we want to use
    sample_with_replacement=True,
    seed=42,
    shuffle=True,
)

# Notice that it only has 4 unique scenes, since we set the dataset_size to 4
print(set(data_loader.dataset))

data_files = next(iter(data_loader))

data_files[0]

# Pass the data_loader to the environment
env = GPUDriveTorchEnv(
    config=EnvConfig(),
    data_loader=data_loader,
    max_cont_agents=64,
    device="cpu",
)


# ### Deep dive: What is inside a traffic scenario? 🤔🔬

# Though every scenario in the WOMD is unique, they all share the same basic data structure. Traffic scenarios are essentially dictionaries, which you can inspect using tools like [JSON Formatter](https://jsonformatter.org/json-viewer). We'll also look at one in this notebook. In a nutshell, traffic scenarios contain a few key elements:
#
# - **Road map**: The layout and structure of the roads.
# - **Human driving (expert) demonstrations**: Examples of human driving behavior.
# - **Road objects**: Elements such as stop signs and other traffic signals.

# Take an example scene
data_path = "data/processed/examples/tfrecord-00000-of-01000_4.json"

with open(data_path) as file:
    traffic_scene = json.load(file)

traffic_scene.keys()


#
# We will show you how to render a scene in ⏭️ tutorial `03`, which introduces the gym environment wrapper. Let's first take a closer look at the data structure.

# ### Global Overview
#
# A traffic scene includes the following key elements:
#
# - **`name`**: The name of the traffic scenario.
# - **`scenario_id`**: Unique identifier for every scenario.
# - **`objects`**: Dynamic entities such as vehicles or other moving elements in the scene.
# - **`roads`**: Stationary elements, including road points and fixed objects.
# - **`tl_states`**: Traffic light states (currently not included in processing).
# - **`metadata`**: Additional details about the traffic scenario, such as the index of the self-driving car (SDC) and details for the WOSAC Challenge.

traffic_scene["tl_states"]
traffic_scene["name"]
traffic_scene["metadata"]
traffic_scene["scenario_id"]

value_counts = pd.Series(
    [
        traffic_scene["objects"][idx]["type"]
        for idx in range(len(traffic_scene["objects"]))
    ]
).value_counts()

cmap = plt.get_cmap("tab20")
colors = [cmap(i) for i in range(len(value_counts))]

value_counts.plot(kind="bar", rot=45, color=colors)
plt.title(
    f"Distribution of road objects in traffic scene. Total # objects: {len(traffic_scene['objects'])}"
)
plt.tight_layout()

output_file = "traffic_scene_distribution.png"
plt.savefig(output_file, dpi=150, bbox_inches="tight")
print(f"Saved to {output_file}")
plt.close()

# This traffic scenario only contains vehicles and pedestrians, some scenes have cyclists as well.

road_value_counts = pd.Series(
    [traffic_scene["roads"][idx]["type"] for idx in range(len(traffic_scene["roads"]))]
).value_counts()

road_colors = [cmap(i) for i in range(len(road_value_counts))]

road_value_counts.plot(kind="bar", rot=45, color=road_colors)
plt.title(
    f"Distribution of road points in traffic scene. Total # points: {len(traffic_scene['roads'])}"
)
plt.show()

# ### In-Depth: Road Objects
#
# This is a list of different road objects in the traffic scene. For each road object, we have information about its position, velocity, size, in which direction it's heading, whether it's a valid object, the type, and the final position of the vehicle.

# Take the first object
idx = 0

# For each object, we have this information:
traffic_scene["objects"][idx].keys()

# Position contains the (x, y) coordinates for the vehicle at every time step
print(json.dumps(traffic_scene["objects"][idx]["position"][:10], indent=4))

# Width and length together make the size of the object, and is used to see if there is a collision
traffic_scene["objects"][idx]["width"], traffic_scene["objects"][idx]["length"]

# An object's heading refers to the direction it is pointing or moving in. The default coordinate system in Nocturne is right-handed, where the positive x and y axes point to the right and downwards, respectively. In a right-handed coordinate system, 0 degrees is located on the x-axis and the angle increases counter-clockwise.
#
# Because the scene is created from the viewpoint of an ego driver, there may be instances where the heading of certain vehicles is not available. These cases are represented by the value `-10_000`, to indicate that these steps should be filtered out or are invalid.

# Heading is the direction in which the vehicle is pointing
plt.plot(traffic_scene["objects"][idx]["heading"])
plt.xlabel("Time step")
plt.ylabel("Heading")
plt.show()

# Velocity shows the velocity in the x- and y- directions
print(json.dumps(traffic_scene["objects"][idx]["velocity"][:10], indent=4))

# Valid indicates if the state of the vehicle was observed for each timepoint
plt.xlabel("Time step")
plt.ylabel("IS VALID")
plt.plot(traffic_scene["objects"][idx]["valid"], "_", lw=5)
plt.show()

# Each object has a goalPosition, an (x, y) position within the scene
traffic_scene["objects"][idx]["goalPosition"]

# Finally, we have the type of the vehicle
traffic_scene["objects"][idx]["type"]


# ### In-Depth: Road Points
#
# Road points are static objects in the scene.

traffic_scene["roads"][idx].keys()

# This point represents the edge of a road
traffic_scene["roads"][idx]["type"]

# Geometry contains the (x, y) position(s) for a road point
# Note that this will be a list for road lanes and edges but a single (x, y) tuple for stop signs and alike
print(json.dumps(traffic_scene["roads"][idx]["geometry"][:10], indent=4))
