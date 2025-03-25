import subprocess
import inspect
import time
import os
import json
import numpy as np
from statistics import mean, stdev
from tqdm import tqdm
from collections import Counter

from CybORG import CybORG, CYBORG_VERSION
from CybORG.Agents import B_lineAgent, SleepAgent
from CybORG.Agents.SimpleAgents.Meander import RedMeanderAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2
from Agents.MainAgent import MainAgent
import random

# Configuration
agent_name = 'Blue'
random.seed(0)
# Number of trajectories is now defined in main function
NUM_STEPS = [30]  # Only use 30 steps length
RED_AGENTS = [RedMeanderAgent]  # Only use RedMeanderAgent

# Create Data directory if it doesn't exist
DATA_DIR = "Data/meander_only"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Add debug logging function
def debug_log(message):
    """Print debug message with timestamp"""
    print(f"[DEBUG {time.strftime('%H:%M:%S')}] {message}")

# Track which agent is loaded
agent_counter = Counter()

# Monkey patch MainAgent to track agent selection
original_load_meander = MainAgent.load_meander
original_load_bline = MainAgent.load_bline
original_load_sleep = MainAgent.load_sleep
original_get_action = MainAgent.get_action

def debug_load_meander(self):
    # debug_log("AGENT SELECTION: Loading MEANDER agent")
    agent_counter['meander'] += 1
    return original_load_meander(self)

def debug_load_bline(self):
    # debug_log("AGENT SELECTION: Loading BLINE agent")
    agent_counter['bline'] += 1
    return original_load_bline(self)

def debug_load_sleep(self):
    # debug_log("AGENT SELECTION: Loading SLEEP agent")
    agent_counter['sleep'] += 1
    return original_load_sleep(self)

def debug_get_action(self, observation, action_space=None):
    # Only track fingerprinting once per trajectory
    if not hasattr(self, 'step_counter'):
        self.step_counter = 0
        self.fingerprint_logged = False
    
    # Log fingerprinting decision
    if len(self.start_actions) == 0 and not self.agent_loaded and not self.fingerprint_logged:
        meander_check = np.sum(self.scan_state) == 3
        bline_check = np.sum(self.scan_state) == 2
        self.fingerprint_info = {
            'meander_check': meander_check,
            'bline_check': bline_check
        }
        self.fingerprint_logged = True
        
        # Force agent to be meander only - override any fingerprinting
        if not self.agent_loaded:
            # Skip the fingerprinting check and always load meander
            self.agent = self.load_meander()
            self.agent_loaded = True
    
    # Call original method
    action = original_get_action(self, observation, action_space)
    
    # Track step count
    self.step_counter += 1
    return action

# Apply the monkey patches
MainAgent.load_meander = debug_load_meander
MainAgent.load_bline = debug_load_bline
MainAgent.load_sleep = debug_load_sleep
MainAgent.get_action = debug_get_action

def wrap(env):
    return ChallengeWrapper2(env=env, agent_name='Blue')

def get_git_revision_hash() -> str:
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    except:
        return "Not using git"

def save_trajectory(trajectory, file_path, agent):
    """Save a trajectory to a JSON file with enhanced metadata"""
    # Add fingerprinting information if available
    if hasattr(agent, 'fingerprint_info'):
        trajectory['metadata']['fingerprinting'] = agent.fingerprint_info
    
    # Add agent allocation information
    trajectory['metadata']['agent_counters'] = dict(agent_counter)
    
    with open(file_path, 'w') as f:
        json.dump(trajectory, f, indent=2, default=convert_to_serializable)

def convert_to_serializable(obj):
    """Convert non-serializable objects to serializable format"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    else:
        return str(obj)

def generate_trajectories(max_trajectories=100):
    """Generate and save trajectories using the trained agent"""
    cyborg_version = CYBORG_VERSION
    scenario = 'Scenario2'
    commit_hash = get_git_revision_hash()
    
    # Get path to scenario
    path = str(inspect.getfile(CybORG))
    path = path[:-10] + f'/Shared/Scenarios/{scenario}.yaml'
    
    print(f'Using CybORG v{cyborg_version}, {scenario}')
    
    # Generate trajectories for different configurations
    for num_steps in NUM_STEPS:
        for red_agent in RED_AGENTS:
            print(f'Generating trajectories for red agent {red_agent.__name__} with {num_steps} steps')
            
            # Create a subdirectory for this configuration
            config_dir = os.path.join(DATA_DIR, f"{red_agent.__name__}_{num_steps}steps")
            if not os.path.exists(config_dir):
                os.makedirs(config_dir)
            
            # Reset agent counter for this configuration
            agent_counter.clear()
            
            # Track rewards for progress bar
            all_rewards = []
            
            # Create progress bar
            pbar = tqdm(range(max_trajectories), desc=f"{red_agent.__name__}_{num_steps}steps")
            
            for i in pbar:
                # Create a fresh agent for each trajectory
                agent = MainAgent()
                # debug_log(f"Starting trajectory {i} with red agent {red_agent.__name__}")
                
                # Initialize environment
                cyborg = CybORG(path, 'sim', agents={'Red': red_agent})
                wrapped_cyborg = wrap(cyborg)
                observation = wrapped_cyborg.reset()
                action_space = wrapped_cyborg.get_action_space(agent_name)
                
                # Initialize trajectory data
                trajectory = {
                    'metadata': {
                        'cyborg_version': cyborg_version,
                        'scenario': scenario,
                        'commit_hash': commit_hash,
                        'red_agent': red_agent.__name__,
                        'steps': num_steps,
                        'trajectory_id': i
                    },
                    'steps': []
                }
                
                # Generate episode
                episode_reward = 0
                for j in range(num_steps):
                    # Get action from agent
                    action = agent.get_action(observation, action_space)
                    
                    # Record pre-step state
                    blue_action = str(action)
                    
                    # Take step in environment
                    next_observation, reward, done, info = wrapped_cyborg.step(action)
                    
                    # Record post-step information
                    red_action = str(cyborg.get_last_action('Red'))
                    blue_action_executed = str(cyborg.get_last_action('Blue'))
                    
                    # Add step data to trajectory
                    step_data = {
                        'step': j,
                        'observation': observation,
                        'action': blue_action,
                        'action_executed': blue_action_executed,
                        'red_action': red_action,
                        'reward': reward,
                        'next_observation': next_observation,
                        'done': done
                    }
                    trajectory['steps'].append(step_data)
                    
                    # Update for next step
                    observation = next_observation
                    episode_reward += reward
                
                # Add total episode reward to metadata
                trajectory['metadata']['total_reward'] = episode_reward
                all_rewards.append(episode_reward)
                
                # Update progress bar with mean reward and agent allocation
                mean_reward = mean(all_rewards) if all_rewards else 0
                agent_stats = f"B:{agent_counter['bline']} M:{agent_counter['meander']} S:{agent_counter['sleep']}"
                pbar.set_postfix({
                    'mean_reward': f'{mean_reward:.2f}',
                    'agents': agent_stats
                })
                
                # Save trajectory
                file_path = os.path.join(config_dir, f"trajectory_{i}.json")
                save_trajectory(trajectory, file_path, agent)
                
                # Reset agent for next episode
                agent.end_episode()
                # debug_log(f"Completed trajectory {i} with reward {episode_reward}")
            
            # Print final statistics
            print(f"Completed {max_trajectories} trajectories for {red_agent.__name__} with {num_steps} steps")
            print(f"Mean reward: {mean(all_rewards):.2f}, Std dev: {stdev(all_rewards):.2f}")
            print(f"Agent allocation: Bline: {agent_counter['bline']}, Meander: {agent_counter['meander']}, Sleep: {agent_counter['sleep']}")

if __name__ == "__main__":
    print("Starting trajectory generation...")
    # Set number of trajectories to generate
    MAX_TRAJECTORIES = 2500
    generate_trajectories(MAX_TRAJECTORIES)
    print("Trajectory generation complete!")
