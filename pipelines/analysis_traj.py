import hydra
import d4rl
import gym
from utils import set_seed
import matplotlib.pyplot as plt
import numpy as np  # Make sure to import numpy
from matplotlib.collections import LineCollection
from sklearn.cluster import KMeans
import time

def calc_k_sim():
    task_name = "antmaze-large-diverse-v2"
    save_path = f'results/diffuser_d4rl_antmaze_rnd/antmaze-large-diverse-v2/rnd_100_1/'
    # rnd0_save_path = f'results/diffuser_d4rl_antmaze_rnd/antmaze-medium-play-v2/rnd_0/'
    # rnd10_save_path = f'results/diffuser_d4rl_antmaze_rnd/antmaze-medium-play-v2/rnd_10_90000/'
    
    obs_path = save_path + f"episode_{0}_obs.npy"
    act_path = save_path + f"episode_{0}_act.npy"
    obs = np.load(obs_path).astype(np.float64)
    act = np.load(act_path).astype(np.float64)
    
    # Load the D4RL dataset
    env = gym.make(task_name)
    dataset = env.get_dataset()
    observations = dataset["observations"].astype(np.float64)
    actions = dataset["actions"].astype(np.float64)
    
    # Perform K-means clustering on the observations
    # Number of clusters
    n_clusters = 50
    
    start_time = time.time()

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(observations[:, :])
    end_time = time.time()
    print(f"K-means clustering took {end_time - start_time} seconds")
    
    # Get the cluster labels for each observation in the dataset
    cluster_labels = kmeans.labels_
    
    # Create a dictionary to store indices of observations in each cluster
    cluster_indices = {}
    for i in range(n_clusters):
        cluster_indices[i] = np.where(cluster_labels == i)[0]
    
    # Print the number of observations in each cluster
    for i in range(n_clusters):
        print(f"Cluster {i}: {len(cluster_indices[i])} observations")
    
    
    distance_list = []
    for i in range(50):
    # Find the closest cluster for each point in the trajectory obs[0]
        trajectory = obs[i, :, :] 

        # Predict the cluster for each point in the trajectory
        trajectory_clusters = kmeans.predict(trajectory[:, :])

        # Print the cluster assignment for each time step
        # print("\nTrajectory Cluster Assignments:")
        # for t, cluster in enumerate(trajectory_clusters):
        #     print(f"Time step {t}: Cluster {cluster}")
            
        nearest_set_size = 50

        for t, cluster in enumerate(trajectory_clusters):
            # Get all observations in the current cluster
            cluster_obs_indices = cluster_indices[cluster]
            cluster_observations = observations[cluster_obs_indices]
            
            # Calculate distances from the current trajectory point to all observations in the cluster
            current_point = trajectory[t]
            distances = np.linalg.norm(cluster_observations - current_point, axis=1)
            
            # Get indices of the nearest_set_size closest points
            nearest_indices = np.argsort(distances)[:nearest_set_size]
            
            # Get the actual indices in the original observations array
            original_indices = cluster_obs_indices[nearest_indices]
            
            # Get the distances of these nearest points
            # nearest_distances = distances[nearest_indices]
            
            # print(f"Time step {t} (Cluster {cluster}):")
            # print(f"  Nearest {nearest_set_size} points indices: {original_indices}")
            # print(f"  Distances: {nearest_distances}")
            
            dataset_obs_act = np.concatenate((observations[nearest_indices], actions[original_indices]), axis=1)
            traj_obs_act = np.concatenate((trajectory[t], act[i,t]), axis=0)
            total_dinstance = np.linalg.norm(dataset_obs_act - traj_obs_act, axis=1)
            distance_list.append(min(total_dinstance))
        
        # Optionally, you can store these results for later use
        # For example, in a dictionary keyed by time step
        # nearest_points_by_timestep[t] = {
        #     'indices': original_indices,
        #     'distances': nearest_distances
        # }
    
    gamma = 10.0
    distance_list = np.array(distance_list)
    distance_list = np.minimum(1.0,gamma/distance_list)
    distance_list = distance_list.sum() / distance_list.shape[0]
    print(distance_list)
    
    
    
    # # Visualize the clusters and trajectory
    # plt.figure(figsize=(10, 8))
    
    # # Plot all observations colored by cluster
    # for i in range(n_clusters):
    #     cluster_obs = observations[cluster_labels == i, :2]
    #     plt.scatter(cluster_obs[:, 0], cluster_obs[:, 1], alpha=0.3, label=f'Cluster {i}' if i < 5 else "")
    
    # # Plot the trajectory with a different color and larger markers
    # plt.plot(trajectory[:, 0], trajectory[:, 1], 'k-', linewidth=2)
    # plt.scatter(trajectory[:, 0], trajectory[:, 1], c=trajectory_clusters, cmap='viridis', 
    #             s=100, edgecolors='black', zorder=5)
    
    # plt.title('K-means Clustering of Observations and Trajectory')
    # plt.xlabel('X Coordinate')
    # plt.ylabel('Y Coordinate')
    # plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    # plt.grid(True)
    # plt.show()
    
    return cluster_indices, trajectory_clusters

def draw_ratio():
    plt.rc('font', family='Times New Roman')
    # Success rates for different RND coefficients
    rnd_coefficients = [0.1, 1, 5, 10, 20, 50, 100, 200, 1000, 10000]  # Adjusted to avoid log(0)
    success_rates = [10/50, 12/50, 14/50, 18/50, 13/50, 15/50, 18/50, 17/50, 19/50, 4/50]  # Average success rates from the log

    plt.plot(rnd_coefficients, success_rates, marker='o', label='Success Rate')
    plt.axhline(y=success_rates[0], color='r', linestyle='--', label='λ=0 Success Rate')
    plt.xlabel('RND Coefficient (λ)', fontsize=14)
    plt.ylabel('Success Rate', fontsize=14)
    plt.ylim(0, 0.5)
    plt.xscale('log')  # Set x-axis to logarithmic scale
    # plt.title('Success Rate of AntMaze Medium Play', fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.show()

def draw_map_big():
    RESET = R = 'r'  # Reset position.
    GOAL = G = 'g'  
    HARDEST_MAZE_TEST = [[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, R, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                    [1, 0, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                    [1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1],
                    [1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1],
                    [1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1],
                    [1, 0, 0, 1, 0, 0, 0, 1, 0, G, 0, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]]
    maze_scale = 4.0
    block_size = 3.7
    
    # Get maze dimensions
    height = len(HARDEST_MAZE_TEST)
    width = len(HARDEST_MAZE_TEST[0])
    
    # Create figure and axis for two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    for ax in axs:
        ax.set_axis_off()
        ax.grid(True)
        # Draw each cell in the maze
        for i in range(height):
            for j in range(width):
                cell = HARDEST_MAZE_TEST[i][j]
                
                # Calculate cell position relative to the reset position (origin)
                x = (j - 1) * maze_scale  # Adjusted to center the maze
                y = (i - 1) * maze_scale  # Adjusted to center the maze
                
                if cell == 1:  # Wall
                    rect = plt.Rectangle((x - maze_scale/2, y - maze_scale/2), maze_scale, maze_scale, 
                                        color='black', fill=True)
                    ax.add_patch(rect)
                elif cell == R:  # Reset position
                    rect = plt.Rectangle((x - block_size/2, y - block_size/2), block_size, block_size, 
                                        color='lightgray', fill=True)
                    ax.add_patch(rect)
                    ax.plot(x, y, 'bo', markersize=10, zorder=5)  # Mark reset position with red dot
                elif cell == G:  # Goal position
                    rect = plt.Rectangle((x - block_size/2, y - block_size/2), block_size, block_size, 
                                        color='lightgray', fill=True)
                    ax.add_patch(rect)
                    ax.plot(32.23, 24.27, 'go', markersize=10, zorder=5)  # Mark goal position with green dot
                else:  # Empty space
                    rect = plt.Rectangle((x - block_size/2, y - block_size/2), block_size, block_size, 
                                        color='lightgray', fill=True)
                    ax.add_patch(rect)

    
    # Load trajectory data
    save_path = f'results/diffuser_d4rl_antmaze_rnd/antmaze-large-play-v2/rnd_10/'
    obs_path = save_path + f"episode_{0}_obs.npy"
    act_path = save_path + f"episode_{0}_act.npy"
    obs = np.load(obs_path)
    act = np.load(act_path)
    path = obs[0,:500:3,:2]
    
    # Plot trajectory with color gradient from light red to dark red
    points = np.array([path[:, 0], path[:, 1]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Use the YlOrRd colormap
    cmap = plt.cm.Reds
    norm = plt.Normalize(-len(path)/3, len(path)*1.1)

    # Add a scatter plot for the points with the same colormap
    scatter = axs[0].scatter(path[:, 0], path[:, 1], c=np.arange(len(path)), 
                            cmap=cmap, norm=norm, s=30, zorder=3)
    
    save_path = f'results/diffuser_d4rl_antmaze_rnd/antmaze-large-play-v2/rnd_0/'
    obs_path = save_path + f"episode_{0}_obs.npy"
    act_path = save_path + f"episode_{0}_act.npy"
    obs = np.load(obs_path)
    act = np.load(act_path)
    path = obs[3,::3,:2]
    
    # Plot trajectory with color gradient from light red to dark red
    points = np.array([path[:, 0], path[:, 1]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Use the YlOrRd colormap
    cmap = plt.cm.Reds
    norm = plt.Normalize(-len(path)/3, len(path)*1.1)

    # Add a scatter plot for the points with the same colormap
    scatter = axs[1].scatter(path[:, 0], path[:, 1], c=np.arange(len(path)), 
                            cmap=cmap, norm=norm, s=30, zorder=3)
    
    
    # Set aspect ratio to be equal
    axs[0].set_aspect('equal')
    axs[1].set_aspect('equal')
    
    # axs[0].set_title('Curiosity-Diffuser', fontsize=12, loc='center', pad=10)
    # axs[1].set_title('Diffuser', fontsize=12, loc='center', pad=10)
    
    plt.show()

def draw_map_medium():
    RESET = R = 'r'  # Reset position.
    GOAL = G = 'g'  
    BIG_MAZE_TEST = [[1, 1, 1, 1, 1, 1, 1, 1],
                [1, R, 0, 1, 1, 0, 0, 1],
                [1, 0, 0, 1, 0, 0, 0, 1],
                [1, 1, 0, 0, 0, 1, 1, 1],
                [1, 0, 0, 1, 0, 0, 0, 1],
                [1, 0, 1, 0, 0, 1, 0, 1],
                [1, 0, 0, 0, 1, 0, G, 1],
                [1, 1, 1, 1, 1, 1, 1, 1]]
    maze_scale = 4.0
    block_size = 3.7
    
    # Get maze dimensions
    height = len(BIG_MAZE_TEST)
    width = len(BIG_MAZE_TEST[0])
    
    # Create figure and axis for two subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    
    for ax in axs:
        ax.set_axis_off()
        ax.grid(True)
        # Draw each cell in the maze
        for i in range(height):
            for j in range(width):
                cell = BIG_MAZE_TEST[i][j]
                
                # Calculate cell position relative to the reset position (origin)
                x = (j - 1) * maze_scale  # Adjusted to center the maze
                y = (i - 1) * maze_scale  # Adjusted to center the maze
                
                if cell == 1:  # Wall
                    rect = plt.Rectangle((x - maze_scale/2, y - maze_scale/2), maze_scale, maze_scale, 
                                        color='black', fill=True)
                    ax.add_patch(rect)
                elif cell == R:  # Reset position
                    rect = plt.Rectangle((x - block_size/2, y - block_size/2), block_size, block_size, 
                                        color='lightgray', fill=True)
                    ax.add_patch(rect)
                    ax.plot(x, y, 'bo', markersize=10, zorder=5)  # Mark reset position with red dot
                elif cell == G:  # Goal position
                    rect = plt.Rectangle((x - block_size/2, y - block_size/2), block_size, block_size, 
                                        color='lightgray', fill=True)
                    ax.add_patch(rect)
                    ax.plot(20.41, 20.87, 'go', markersize=10, zorder=5)  # Mark goal position with green dot
                else:  # Empty space
                    rect = plt.Rectangle((x - block_size/2, y - block_size/2), block_size, block_size, 
                                        color='lightgray', fill=True)
                    ax.add_patch(rect)

    
    # Load trajectory data
    save_path = f'results/diffuser_d4rl_antmaze_rnd/antmaze-medium-play-v2/rnd_10_90000/'
    obs_path = save_path + f"episode_{0}_obs.npy"
    act_path = save_path + f"episode_{0}_act.npy"
    obs = np.load(obs_path)
    act = np.load(act_path)
    path = obs[4,:390:3,:2]
    
    # Plot trajectory with color gradient from light red to dark red
    points = np.array([path[:, 0], path[:, 1]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Use the YlOrRd colormap
    cmap = plt.cm.Reds
    norm = plt.Normalize(-len(path)/3, len(path)*1.1)

    # Add a scatter plot for the points with the same colormap
    scatter = axs[0].scatter(path[:, 0], path[:, 1], c=np.arange(len(path)), 
                            cmap=cmap, norm=norm, s=30, zorder=3)
    
    save_path = f'results/diffuser_d4rl_antmaze_rnd/antmaze-medium-play-v2/rnd_0/'
    obs_path = save_path + f"episode_{0}_obs.npy"
    act_path = save_path + f"episode_{0}_act.npy"
    obs = np.load(obs_path)
    act = np.load(act_path)
    path = obs[5,::3,:2]
    
    # Plot trajectory with color gradient from light red to dark red
    points = np.array([path[:, 0], path[:, 1]]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    
    # Use the YlOrRd colormap
    cmap = plt.cm.Reds
    norm = plt.Normalize(-len(path)/3, len(path)*1.1)

    # Add a scatter plot for the points with the same colormap
    scatter = axs[1].scatter(path[:, 0], path[:, 1], c=np.arange(len(path)), 
                            cmap=cmap, norm=norm, s=30, zorder=3)
    
    
    # Set aspect ratio to be equal
    axs[0].set_aspect('equal')
    axs[1].set_aspect('equal')
    
    # axs[0].set_title('Curiosity-Diffuser', fontsize=12, loc='center', pad=10)
    # axs[1].set_title('Diffuser', fontsize=12, loc='center', pad=10)
    
    plt.show()
    

if __name__ == "__main__":
    # draw_ratio()
    # draw_map_medium()
    # calc_k_sim()
    draw_map_big()