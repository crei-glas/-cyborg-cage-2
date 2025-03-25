# checkout https://github.com/geekyutao/PyTorch-PPO/blob/master/PPO_discrete.py

import torch
import numpy as np
import os
import glob
import argparse
import wandb
from CybORG import CybORG
from CybORG.Agents import RedMeanderAgent, B_lineAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2
import inspect
from Agents.PPOAgent import PPOAgent
import random
from tqdm import tqdm

PATH = str(inspect.getfile(CybORG))
PATH = PATH[:-10] + '/Shared/Scenarios/Scenario2.yaml'
device = torch.device("mps" if torch.backends.mps.is_available() else 
                     ("cuda:0" if torch.cuda.is_available() else "cpu"))
print(f"Using device: {device}")


def find_latest_checkpoint(ckpt_folder):
    """Find the latest checkpoint in the folder based on episode number."""
    checkpoints = glob.glob(os.path.join(ckpt_folder, '*.pth'))
    if not checkpoints:
        return None, 0
    
    # Extract episode numbers from filenames
    episodes = []
    for ckpt in checkpoints:
        basename = os.path.basename(ckpt).split('.')[0]
        # Handle the case where the checkpoint is named "model.pth"
        if basename == "model":
            # Assume this is a final model, give it a high episode number
            episodes.append(float('inf'))  # Use infinity to represent final model
        # Handle the case where the checkpoint is named "best.pth"
        elif basename == "best":
            # Assume this is the best model, give it a very high episode number
            episodes.append(float('inf') - 1)  # Just below infinity
        else:
            try:
                episodes.append(int(basename))
            except ValueError:
                # Skip files with non-integer names
                continue
    
    if not episodes:
        return None, 0
    
    # Find the latest episode
    latest_episode = max(episodes)
    
    # Handle the cases where the latest episode is infinity or infinity-1
    if latest_episode == float('inf'):
        latest_checkpoint = os.path.join(ckpt_folder, 'model.pth')
        # Use a very large number to represent completion
        latest_episode = 999999
    elif latest_episode == float('inf') - 1:
        latest_checkpoint = os.path.join(ckpt_folder, 'best.pth')
        # Use a large number to represent best model
        latest_episode = 999998
    else:
        latest_checkpoint = os.path.join(ckpt_folder, f'{latest_episode}.pth')
    
    return latest_checkpoint, latest_episode


def train(env, input_dims, action_space,
          max_episodes, max_timesteps, update_timestep, K_epochs, eps_clip,
          gamma, lr, betas, ckpt_folder, print_interval=10, save_interval=100, 
          start_actions=[], approach_name="default", bc_model_path=None):

    # Use a lower learning rate if we're warm-starting
    actual_lr = lr * 0.1 if bc_model_path else lr
    
    # Get model name from checkpoint folder path
    model_name = os.path.basename(ckpt_folder)
    
    # Check if training is already complete for this model
    latest_checkpoint, latest_episode = find_latest_checkpoint(ckpt_folder)
    
    if latest_episode >= max_episodes:
        print(f"Training already complete for {model_name} ({latest_episode}/{max_episodes} episodes)")
        return True  # Training complete
    
    # IMPORTANT: Load BC model first to get dimensions
    bc_input_dims = input_dims
    bc_action_space_size = len(action_space)
    
    # If we're warm-starting, load the BC model to get its dimensions
    if bc_model_path:
        print(f"Loading pre-trained BC model from {bc_model_path}")
        
        # Load model state dict to inspect its dimensions
        bc_state_dict = torch.load(bc_model_path, map_location=device)
        
        # Extract input dimensions from first layer weight
        if "actor.0.weight" in bc_state_dict:
            bc_input_dims = bc_state_dict["actor.0.weight"].shape[1]
            print(f"BC model input dimensions: {bc_input_dims}")
        
        # Extract action space size from last layer
        if "actor.4.weight" in bc_state_dict:
            bc_action_space_size = bc_state_dict["actor.4.weight"].shape[0]
            print(f"BC model action space size: {bc_action_space_size}")
        
        # Warning if dimensions don't match
        if bc_input_dims != input_dims:
            print(f"WARNING: BC input dims ({bc_input_dims}) don't match environment dims ({input_dims})")
        
        if bc_action_space_size != len(action_space):
            print(f"WARNING: BC action space size ({bc_action_space_size}) doesn't match the provided action space ({len(action_space)})")
    
    # Initialize agent with the BC model dimensions
    agent = PPOAgent(bc_input_dims, action_space, actual_lr, betas, gamma, K_epochs, eps_clip, start_actions=start_actions)
    
    # Now load the BC model state dict if provided
    if bc_model_path:
        try:
            agent.policy.load_state_dict(bc_state_dict)
            print("Successfully loaded pre-trained BC model weights")
        except Exception as e:
            print(f"Failed to load pre-trained BC model: {e}")
            # Continue with randomly initialized weights
    
    # Initialize wandb
    run_id_file = os.path.join(ckpt_folder, "wandb_run_id.txt")
    run_id = None
    
    # Check if we have a previous run ID
    if os.path.exists(run_id_file):
        with open(run_id_file, "r") as f:
            run_id = f.read().strip()
            print(f"Resuming wandb run: {run_id}")
    
    # Initialize wandb with resume capability
    run = wandb.init(
        project="cyborg-cage",
        name=f"{approach_name}-{model_name}",
        id=run_id,
        resume="allow",
        config={
            "model": model_name,
            "approach": approach_name,
            "max_episodes": max_episodes,
            "max_timesteps": max_timesteps,
            "update_timestep": update_timestep,
            "K_epochs": K_epochs,
            "eps_clip": eps_clip,
            "gamma": gamma,
            "lr": lr,
            "betas": betas,
            "device": str(device),
            "action_space": action_space,
            "input_dims": input_dims,
            "warm_started": bc_model_path is not None,
            "bc_model_path": bc_model_path
        }
    )
    
    # Save the run ID for future resuming
    if not os.path.exists(run_id_file):
        with open(run_id_file, "w") as f:
            f.write(run.id)
    
    # Load checkpoint if exists
    start_episode = 1
    # If we have a past checkpoint, load that
    if latest_checkpoint:
        print(f"Resuming training from episode {latest_episode}")
        checkpoint = torch.load(latest_checkpoint, weights_only=False)
        agent.policy.load_state_dict(checkpoint['model_state_dict'])
        agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Restore other training state if available
        if 'reward_window' in checkpoint:
            reward_window = checkpoint['reward_window']
        if 'time_step' in checkpoint:
            time_step = checkpoint['time_step']
        
        start_episode = latest_episode + 1
        
        # Log to wandb that we're resuming
        wandb.log({"resumed_from_episode": latest_episode})
    # Otherwise, if we have a BC model, load that
    elif bc_model_path:
        print(f"Loading pre-trained BC model from {bc_model_path}")
        # Load the BC model state dict
        bc_state_dict = torch.load(bc_model_path, map_location=device)
        
        # If it's a full checkpoint (like from pretrain.py), extract just the model state dict
        if isinstance(bc_state_dict, dict) and 'model_state_dict' in bc_state_dict:
            bc_state_dict = bc_state_dict['model_state_dict']
        
        # Load the state dict into our agent's policy
        agent.policy.load_state_dict(bc_state_dict)
        
        # Log to wandb that we're warm-starting
        wandb.log({"warm_started_from": bc_model_path})
    else:
        print(f"Starting new training for {model_name}")
    
    # Create a sliding window for reward tracking
    reward_window = []
    time_step = 0
    
    # Create tqdm progress bar for episodes with dynamic_ncols=True
    remaining_episodes = max_episodes - start_episode + 1
    pbar = tqdm(range(start_episode, max_episodes + 1), 
                desc=f"Training {model_name}", 
                initial=start_episode-1, 
                total=max_episodes,
                dynamic_ncols=True,  # Automatically adjust to terminal width
                smoothing=0.1)       # Smoother progress bar updates
    
    # Training metrics
    episode_lengths = []
    training_updates = 0
    
    # Add after agent initialization
    if bc_model_path:
        # Start with very low learning rate and gradually increase
        for param_group in agent.optimizer.param_groups:
            param_group['lr'] = lr * 0.01  # Start at 1% of normal learning rate
        
        # Create a simple linear scheduler to increase learning rate over first 1000 episodes
        def lr_lambda(episode):
            if episode < 1000:
                return 0.01 + (1.0 - 0.01) * (episode / 1000)  # Linear increase
            else:
                return 1.0
        
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(agent.optimizer, lr_lambda)
    
    for i_episode in pbar:
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for t in range(max_timesteps):
            time_step += 1
            episode_length += 1
            action = agent.get_action(state)
            state, reward, done, _ = env.step(action)
            agent.store(reward, done)
            
            episode_reward += reward

            if bc_model_path and i_episode < 1000:
                # Reduce update frequency for first 1000 episodes when warm-starting
                effective_update_timestep = update_timestep * 2
            else:
                effective_update_timestep = update_timestep

            if time_step % effective_update_timestep == 0:
                agent.train()
                agent.clear_memory()
                time_step = 0
                training_updates += 1
                wandb.log({"training_updates": training_updates}, step=i_episode)

            if done:
                break
                
        agent.end_episode()
        episode_lengths.append(episode_length)
        
        # Update sliding window of rewards
        reward_window.append(episode_reward)
        if len(reward_window) > print_interval:
            reward_window.pop(0)  # Remove oldest reward to maintain window size
        
        # Calculate average reward over the sliding window
        avg_reward = sum(reward_window) / len(reward_window)
        
        # Update progress bar with current reward
        pbar.set_postfix({"Avg Reward:": f"{avg_reward:.2f}"})
        
        # Log metrics to wandb
        wandb.log({
            "episode": i_episode,
            "episode_reward": episode_reward,
            "episode_length": episode_length,
            "avg_reward": avg_reward,
            "reward_window_size": len(reward_window),
        }, step=i_episode)

        if i_episode % save_interval == 0:
            ckpt = os.path.join(ckpt_folder, '{}.pth'.format(i_episode))
            # Save both model and optimizer state
            torch.save({
                'episode': i_episode,
                'model_state_dict': agent.policy.state_dict(),
                'optimizer_state_dict': agent.optimizer.state_dict(),
                'reward_window': reward_window,  # Save the reward window
                'time_step': time_step,  # Save the timestep counter
            }, ckpt)
            wandb.save(ckpt)
            # print('Checkpoint saved')

        if i_episode % print_interval == 0:
            continue
            # print(f'Episode {i_episode} \t Avg reward: {avg_reward:.2f}')
    
    # Save final model as model.pth when training completes
    final_model_path = os.path.join(ckpt_folder, 'model.pth')
    torch.save(agent.policy.state_dict(), final_model_path)
    wandb.save(final_model_path)
    print(f"Training complete. Final model saved as {final_model_path}")
    
    # Finish wandb run
    wandb.finish()
    
    return True  # Training complete


def train_multiple_models(models_config, common_params, approach_name):
    """Train multiple models sequentially, picking up where each left off."""
    for model_config in models_config:
        model_name = model_config['name']
        red_agent = model_config['red_agent']
        bc_model_path = model_config.get('bc_model_path', None)
        
        print(f"\n{'='*50}")
        if bc_model_path:
            print(f"Setting up warm-start training for {approach_name}/{model_name} against {red_agent.__name__}")
            print(f"Using pre-trained BC model: {bc_model_path}")
        else:
            print(f"Setting up training for {approach_name}/{model_name} against {red_agent.__name__}")
        print(f"{'='*50}\n")
        
        # Setup checkpoint directory with approach name as parent directory
        ckpt_folder = os.path.join(os.getcwd(), "Models", approach_name, model_name)
        if not os.path.exists(ckpt_folder):
            os.makedirs(ckpt_folder)
        
        # Setup environment with specified red agent
        CYBORG = CybORG(PATH, 'sim', agents={'Red': red_agent})
        env = ChallengeWrapper2(env=CYBORG, agent_name="Blue")
        input_dims = env.observation_space.shape[0]
        
        # Train the model
        training_complete = train(
            env=env, 
            input_dims=input_dims,
            action_space=common_params['action_space'],
            max_episodes=common_params['max_episodes'],
            max_timesteps=common_params['max_timesteps'],
            update_timestep=common_params['update_timesteps'],
            K_epochs=common_params['K_epochs'],
            eps_clip=common_params['eps_clip'],
            gamma=common_params['gamma'],
            lr=common_params['lr'],
            betas=common_params['betas'],
            ckpt_folder=ckpt_folder,
            print_interval=common_params['print_interval'],
            save_interval=common_params['save_interval'],
            start_actions=common_params['start_actions'],
            approach_name=approach_name,
            bc_model_path=bc_model_path
        )
        
        if training_complete:
            print(f"\nTraining complete for {approach_name}/{model_name}")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train PPO agents against different red agents')
    parser.add_argument('--approach', type=str, default='default', 
                        help='Name of the approach/experiment (used for organizing model directories)')
    parser.add_argument('--max_episodes', type=int, default=100000,
                        help='Maximum number of episodes to train for')
    parser.add_argument('--print_interval', type=int, default=50,
                        help='How often to print progress')
    parser.add_argument('--save_interval', type=int, default=200,
                        help='How often to save checkpoints')
    parser.add_argument('--no-wandb', action='store_true',
                        help='Disable wandb logging')
    return parser.parse_args()


if __name__ == '__main__':
    # Parse command line arguments
    args = parse_arguments()
    
    # set seeds for reproducibility
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

    # Define action space and other common parameters
    action_space = [133, 134, 135, 139]  # restore enterprise and opserver
    action_space += [3, 4, 5, 9]  # analyse enterprise and opserver
    action_space += [16, 17, 18, 22]  # remove enterprise and opserer
    action_space += [11, 12, 13, 14]  # analyse user hosts
    action_space += [141, 142, 143, 144]  # restore user hosts
    action_space += [132]  # restore defender
    action_space += [2]  # analyse defender
    action_space += [15, 24, 25, 26, 27]  # remove defender and user hosts
    # Add host IDs for decoy actions
    action_space += [1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008]  # host IDs for decoys

    start_actions = [1004, 1004, 1000]  # user 2 decoy * 2, ent0 decoy

    # Common parameters for all models
    common_params = {
        'action_space': action_space,
        'max_episodes': 50000,
        'max_timesteps': 100,
        'update_timesteps': 20000,
        'K_epochs': 6,
        'eps_clip': 0.2,
        'gamma': 0.99,
        'lr': 0.002,
        'betas': [0.9, 0.990],
        'print_interval': args.print_interval,
        'save_interval': args.save_interval,
        'start_actions': start_actions
    }

    # Define only the meander model to train
    models_config = [
        {
            'name': 'meander',
            'red_agent': RedMeanderAgent,
            'bc_model_path': "Models/bc_meander_fixed/meander/model.pth"  # Use the best model from BC
        }
    ]

    # Use a unique approach name for the warm-started models
    warm_start_approach = "warm_start_meander_only"
    print(f"Starting warm-start training with approach: {warm_start_approach}")
    
    # Initialize wandb - enabled by default
    if args.no_wandb:
        # Disable wandb
        os.environ["WANDB_MODE"] = "disabled"
        print("Weights & Biases logging disabled")
    else:
        print("Weights & Biases logging enabled")
    
    # In the main function, create separate parameters for warm-started models
    warm_start_params = common_params.copy()
    warm_start_params.update({
        'K_epochs': 3,           # Fewer policy updates per batch (was 6)
        'eps_clip': 0.1,         # Smaller policy change allowed (was 0.2)
        'lr': 0.0005,            # Much lower learning rate (was 0.002)
        'update_timesteps': 10000  # More frequent updates (was 20000)
    })

    # Train only the meander model
    train_multiple_models(models_config, warm_start_params, warm_start_approach)