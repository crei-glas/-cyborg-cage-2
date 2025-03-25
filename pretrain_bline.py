# checkout https://github.com/geekyutao/PyTorch-PPO/blob/master/PPO_discrete.py

import torch
import numpy as np
import os
import glob
import argparse
import wandb
import json
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from CybORG import CybORG
from CybORG.Agents import RedMeanderAgent, B_lineAgent
from Wrappers.ChallengeWrapper2 import ChallengeWrapper2
import inspect
from Agents.PPOAgent import PPOAgent
import random
from tqdm import tqdm
from PPO.ActorCritic import ActorCritic

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
    
    # Handle the case where the latest episode is infinity (model.pth)
    if latest_episode == float('inf'):
        latest_checkpoint = os.path.join(ckpt_folder, 'model.pth')
        # Use a very large number to represent completion
        latest_episode = 999999
    else:
        latest_checkpoint = os.path.join(ckpt_folder, f'{latest_episode}.pth')
    
    return latest_checkpoint, latest_episode


class TrajectoryDataset(Dataset):
    """Dataset for loading trajectory data for behavior cloning"""
    
    def __init__(self, data_dir):
        self.observations = []
        self.actions = []
        
        # Load all trajectory files from the directory
        trajectory_files = glob.glob(os.path.join(data_dir, "trajectory_*.json"))
        print(f"Loading {len(trajectory_files)} trajectories from {data_dir}")
        
        # Track action distribution
        action_counts = {}
        
        # Create mapping from decoy actions to host IDs (copied from PPOAgent.all_decoys)
        decoy_to_host = {
            # Enterprise0 decoys (1000)
            55: 1000, 107: 1000, 120: 1000, 29: 1000,
            # Enterprise1 decoys (1001)
            43: 1001,
            # Enterprise2 decoys (1002)
            44: 1002,
            # User1 decoys (1003)
            37: 1003, 115: 1003, 76: 1003, 102: 1003,
            # User2 decoys (1004)
            51: 1004, 116: 1004, 38: 1004, 90: 1004,
            # User3 decoys (1005)
            130: 1005, 91: 1005,
            # User4 decoys (1006)
            131: 1006,
            # Defender decoys (1007)
            54: 1007, 106: 1007, 28: 1007, 119: 1007,
            # Opserver0 decoys (1008)
            126: 1008, 61: 1008, 113: 1008, 35: 1008
        }
        
        # Standard action space
        standard_actions = [2, 3, 4, 5, 9, 11, 12, 13, 14, 15, 16, 17, 18, 22, 
                           24, 25, 26, 27, 132, 133, 134, 135, 139, 141, 142, 143, 144]
        
        # DEBUG: Print the first few trajectories to see their format
        print("Standard actions:", standard_actions)
        print("Decoy actions:", list(decoy_to_host.keys()))
        
        # Keep track of parsing info
        total_steps = 0
        processed_steps = 0
        skipped_steps = 0
        action_formats = set()
        skipped_actions = set()
        trajectories_processed = 0
        trajectories_skipped = 0
        
        # For debugging - check first trajectory
        first_trajectory_processed = False
        
        for file_path in tqdm(trajectory_files, desc="Loading trajectories"):
            try:
                with open(file_path, 'r') as f:
                    trajectory = json.load(f)
                
                # DEBUG: Print the first few actions from the first file
                if file_path == trajectory_files[0]:
                    print("\nSample actions from first trajectory:")
                    for i, step in enumerate(trajectory['steps'][:5]):
                        print(f"Step {i}, Action: {step['action']}, Type: {type(step['action'])}")
                        # Check if this action would be valid
                        action_val = step['action']
                        if isinstance(action_val, str):
                            try:
                                action_val = int(action_val)
                                print(f"  - Converted to int: {action_val}")
                                print(f"  - In standard actions: {action_val in standard_actions}")
                                print(f"  - In decoy actions: {action_val in decoy_to_host}")
                            except (ValueError, TypeError):
                                print(f"  - Could not convert to int")
                
                # Check which blue agent was used if available in metadata
                blue_agent_type = None
                if 'metadata' in trajectory and 'fingerprinting' in trajectory['metadata']:
                    # Track which policy was actually used for this trajectory
                    fingerprint_info = trajectory['metadata']['fingerprinting']
                    if fingerprint_info['meander_check']:
                        blue_agent_type = 'meander'
                    elif fingerprint_info['bline_check']:
                        blue_agent_type = 'bline'
                    else:
                        blue_agent_type = 'sleep'
                
                # For the first few trajectories, print debugging info about filtering
                if trajectories_processed < 5:
                    print(f"\nTrajectory {file_path}:")
                    print(f"  Blue agent type: {blue_agent_type}")
                    print(f"  Number of steps: {len(trajectory['steps'])}")
                
                # TEMPORARILY DISABLE FILTERING FOR DEBUGGING
                # Skip trajectory if it's not bline
                # if blue_agent_type and blue_agent_type != 'bline':
                #     trajectories_skipped += 1
                #     continue
                
                trajectories_processed += 1
                
                # Extract observation-action pairs from each step
                trajectory_steps_processed = 0
                for step in trajectory['steps']:
                    total_steps += 1
                    
                    # Convert observation to numpy array
                    observation = np.array(step['observation'])
                    
                    # Extract action (convert from string to int)
                    action_str = step['action']
                    action_formats.add(type(action_str).__name__)
                    
                    try:
                        # Handle different action string formats
                        if isinstance(action_str, int):
                            # Already an integer
                            action = action_str
                        elif isinstance(action_str, str):
                            if '(' in action_str and ')' in action_str:
                                # Format like "Action(26)"
                                action = int(action_str.split('(')[1].split(')')[0])
                            else:
                                # Try direct conversion if it's just a number as string
                                action = int(action_str)
                        else:
                            print(f"Unknown action type: {type(action_str)}")
                            continue
                        
                        # Process action based on type
                        if action in standard_actions:
                            # Standard action - add directly
                            self.observations.append(observation)
                            self.actions.append(action)
                            action_counts[action] = action_counts.get(action, 0) + 1
                            processed_steps += 1
                            trajectory_steps_processed += 1
                        elif action in decoy_to_host:
                            # Decoy action - map to host ID
                            host_id = decoy_to_host[action]
                            self.observations.append(observation)
                            self.actions.append(host_id)
                            # Track the host ID in action counts
                            action_counts[host_id] = action_counts.get(host_id, 0) + 1
                            processed_steps += 1
                            trajectory_steps_processed += 1
                        else:
                            skipped_steps += 1
                            skipped_actions.add(action)
                            if len(skipped_actions) < 10:  # Limit output
                                print(f"Skipping unknown action: {action}")
                    except (ValueError, IndexError) as e:
                        skipped_steps += 1
                        print(f"Skipping invalid action: {action_str}, Error: {e}")
                
                # For first trajectory, show how many steps we processed
                if file_path == trajectory_files[0]:
                    print(f"\nFirst trajectory: processed {trajectory_steps_processed} of {len(trajectory['steps'])} steps")
                    first_trajectory_processed = True
                
            except Exception as e:
                print(f"Error loading trajectory {file_path}: {e}")
        
        # DEBUG: Print statistics about parsed actions
        print(f"\nParsing statistics:")
        print(f"Total trajectories: {len(trajectory_files)}")
        print(f"Trajectories processed: {trajectories_processed}")
        print(f"Trajectories skipped: {trajectories_skipped}")
        print(f"Total steps found: {total_steps}")
        print(f"Steps processed: {processed_steps}")
        print(f"Steps skipped: {skipped_steps}")
        if total_steps > 0:
            print(f"Processing rate: {processed_steps/total_steps*100:.2f}% of total")
        print(f"Action formats found: {action_formats}")
        print(f"Skipped action values: {skipped_actions}")
        
        # Convert lists to numpy arrays
        if len(self.observations) > 0:
            self.observations = np.array(self.observations, dtype=np.float32)
            self.actions = np.array(self.actions, dtype=np.int64)
            
            # Print dataset statistics
            print(f"Loaded {len(self.observations)} observation-action pairs")
            print(f"Observation shape: {self.observations.shape}")
            print(f"Action distribution:")
            for action, count in sorted(action_counts.items()):
                print(f"  Action {action}: {count} ({count/len(self.actions)*100:.2f}%)")
        else:
            # Create empty arrays to avoid errors
            print("WARNING: No valid observation-action pairs found!")
            self.observations = np.array([], dtype=np.float32).reshape(0, observation.shape[0] if 'observation' in locals() else 100)
            self.actions = np.array([], dtype=np.int64)

    def __len__(self):
        return len(self.observations)

    def __getitem__(self, idx):
        return torch.tensor(self.observations[idx], dtype=torch.float32), torch.tensor(self.actions[idx], dtype=torch.long)


def evaluate_model(model, dataloader, criterion, action_space, device):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for observations, actions in dataloader:
            observations = observations.to(device)
            actions = actions.to(device)
            
            # Convert actions to indices
            action_indices = torch.zeros_like(actions, device=device)
            valid_samples = torch.ones_like(actions, dtype=torch.bool, device=device)
            
            for i, a in enumerate(actions):
                try:
                    action_indices[i] = action_space.index(a.item())
                except ValueError:
                    valid_samples[i] = False
            
            # Skip if no valid samples
            if not torch.any(valid_samples):
                continue
                
            # Filter to valid samples
            if not torch.all(valid_samples):
                observations = observations[valid_samples]
                action_indices = action_indices[valid_samples]
            
            # Get model predictions
            action_probs = model.actor(observations)
            loss = criterion(action_probs, action_indices)
            total_loss += loss.item() * observations.size(0)
            
            # Calculate accuracy
            pred_actions = torch.argmax(action_probs, dim=1)
            correct += (pred_actions == action_indices).sum().item()
            total += observations.size(0)
    
    # Calculate average loss and accuracy
    avg_loss = total_loss / total if total > 0 else float('inf')
    accuracy = correct / total * 100 if total > 0 else 0
    return avg_loss, accuracy


def behavior_cloning(input_dims, action_space, data_dir, ckpt_folder, batch_size=64, num_epochs=100, 
                     lr=0.001, save_interval=10, approach_name="behavior_cloning", 
                     val_ratio=0.2, patience=5):
    """Train a model using behavior cloning on the given dataset."""
    print("\nStarting behavior cloning...")
    print(f"Input dimensions: {input_dims}")
    print(f"Action space: {action_space} (size: {len(action_space)})")
    print(f"Data directory: {data_dir}")
    print(f"Checkpoint folder: {ckpt_folder}")
    print(f"Batch size: {batch_size}, Learning rate: {lr}")
    print(f"Training for {num_epochs} epochs, saving every {save_interval} epochs")
    
    # Load dataset
    full_dataset = TrajectoryDataset(data_dir)
    
    # Make sure we have data
    if len(full_dataset) == 0:
        print("Error: No valid data found for behavior cloning!")
        return False
    
    # Get actual input dimensions from the dataset
    actual_input_dims = full_dataset.observations.shape[1]
    print(f"Dataset observation dimensions: {actual_input_dims}")
    
    # Do NOT add extra dimensions - observations are already padded in dataset
    model_input_dims = actual_input_dims
    print(f"Model input dimensions: {model_input_dims}")
    
    # Create train-validation split
    dataset_size = len(full_dataset)
    val_size = int(val_ratio * dataset_size)
    train_size = dataset_size - val_size
    print(f"Dataset size: {dataset_size} (Train: {train_size}, Val: {val_size})")
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)  # Fixed seed for reproducibility
    )
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    
    # Initialize model
    policy = ActorCritic(model_input_dims, len(action_space)).to(device)
    print(f"Model created with input dims: {model_input_dims}, output dims: {len(action_space)}")
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    # Load latest checkpoint if it exists
    latest_checkpoint, start_epoch = find_latest_checkpoint(ckpt_folder)
    if latest_checkpoint:
        print(f"Loading checkpoint from {latest_checkpoint}")
        policy.load_state_dict(torch.load(latest_checkpoint, map_location=device))
        start_epoch += 1
    else:
        start_epoch = 0
    
    # Setup wandb
    run = wandb.init(
        project="cyborg-behavior-cloning",
        name=f"{approach_name}_bline",
        config={
            "input_dims": model_input_dims,
            "action_space_size": len(action_space),
            "batch_size": batch_size,
            "lr": lr,
            "num_epochs": num_epochs,
            "dataset_size": dataset_size,
        }
    )
    
    # Training loop
    best_val_loss = float('inf')
    best_val_accuracy = 0
    epochs_without_improvement = 0
    pbar = tqdm(range(start_epoch, num_epochs), desc='Training')
    
    for epoch in pbar:
        # Training phase
        policy.train()
        epoch_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        for observations, actions in train_loader:
            # Move observations to device first
            observations = observations.to(device)
            actions = actions.to(device)
            
            # REMOVE double padding - observations are already padded in dataset
            # Use observations directly
            
            # Convert actions to indices in the action space
            action_indices = torch.zeros_like(actions, device=device)
            valid_samples = torch.ones_like(actions, dtype=torch.bool, device=device)
            
            for i, a in enumerate(actions):
                try:
                    action_indices[i] = action_space.index(a.item())
                except ValueError:
                    # Mark invalid samples instead of defaulting to 0
                    valid_samples[i] = False
            
            # Skip batch completely if no valid samples
            if not torch.any(valid_samples):
                continue
                
            # Filter to only use valid samples
            if not torch.all(valid_samples):
                observations = observations[valid_samples]
                action_indices = action_indices[valid_samples]
            
            # Forward pass
            action_probs = policy.actor(observations)
            loss = criterion(action_probs, action_indices)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            epoch_loss += loss.item() * observations.size(0)
            pred_actions = torch.argmax(action_probs, dim=1)
            correct_predictions += (pred_actions == action_indices).sum().item()
            total_predictions += observations.size(0)
        
        # Calculate average training loss and accuracy
        train_loss = epoch_loss / total_predictions if total_predictions > 0 else float('inf')
        train_accuracy = correct_predictions / total_predictions * 100 if total_predictions > 0 else 0
        
        # Validation phase
        val_loss, val_accuracy = evaluate_model(policy, val_loader, criterion, action_space, device)
        
        # Update progress bar
        pbar.set_postfix({'train_loss': f'{train_loss:.4f}', 
                          'train_acc': f'{train_accuracy:.2f}%',
                          'val_loss': f'{val_loss:.4f}', 
                          'val_acc': f'{val_accuracy:.2f}%'})
        
        # Log metrics to wandb
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': train_accuracy,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy
        })
        
        # Save model if it's better
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_val_loss = val_loss
            
            # Save the best model
            best_model_path = os.path.join(ckpt_folder, 'model.pth')
            torch.save(policy.state_dict(), best_model_path)
            print(f"\nSaved best model with val_accuracy: {val_accuracy:.2f}% at epoch {epoch}")
            
            # Reset patience counter
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            
        # Save checkpoint at regular intervals
        if (epoch + 1) % save_interval == 0:
            checkpoint_path = os.path.join(ckpt_folder, f'{epoch}.pth')
            torch.save(policy.state_dict(), checkpoint_path)
        
        # Early stopping
        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping at epoch {epoch} due to no improvement for {patience} epochs")
            break
    
    # Finish wandb run
    wandb.finish()
    
    print(f"\nTraining completed. Best validation accuracy: {best_val_accuracy:.2f}%")
    return True


def train_multiple_models(models_config, common_params, approach_name):
    """Train multiple models using behavior cloning on different datasets."""
    for model_config in models_config:
        model_name = model_config['name']
        data_dir = model_config['data_dir']
        
        print(f"\n{'='*50}")
        print(f"Setting up behavior cloning for {approach_name}/{model_name} using data from {data_dir}")
        print(f"{'='*50}\n")
        
        # Setup checkpoint directory with approach name as parent directory
        ckpt_folder = os.path.join(os.getcwd(), "Models", approach_name, model_name)
        if not os.path.exists(ckpt_folder):
            os.makedirs(ckpt_folder)
        
        # Setup environment to get input dimensions
        CYBORG = CybORG(PATH, 'sim')
        env = ChallengeWrapper2(env=CYBORG, agent_name="Blue")
        input_dims = env.observation_space.shape[0]
        
        # Train the model using behavior cloning
        training_complete = behavior_cloning(
            input_dims=input_dims,
            action_space=common_params['action_space'],
            data_dir=data_dir,
            ckpt_folder=ckpt_folder,
            batch_size=common_params['batch_size'],
            num_epochs=common_params['num_epochs'],
            lr=common_params['lr'],
            save_interval=common_params['save_interval'],
            approach_name=approach_name,
            val_ratio=common_params['val_ratio'],
            patience=common_params['patience']
        )
        
        if training_complete:
            print(f"\nBehavior cloning complete for {approach_name}/{model_name}")


def parse_arguments():
    parser = argparse.ArgumentParser(description='Train agents using behavior cloning on trajectory data')
    parser.add_argument('--approach', type=str, default='behavior_cloning', 
                        help='Name of the approach/experiment (used for organizing model directories)')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs to train for')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--save_interval', type=int, default=10,
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
    
    # Common parameters for behavior cloning
    common_params = {
        'action_space': action_space,
        'batch_size': args.batch_size,
        'num_epochs': 25,  # Changed from 100 to 25
        'lr': 0.001,
        'save_interval': 5,
        'val_ratio': 0.2,   # 20% of data for validation
        'patience': 5       # Stop after 5 epochs without improvement
    }

    # Define models to train - ONLY B_lineAgent
    models_config = [
        {
            'name': 'bline',
            'data_dir': "Data/bline_only/B_lineAgent_30steps"
        }
    ]

    print(f"Starting behavior cloning with approach: {args.approach}")
    
    # Initialize wandb - enabled by default
    if args.no_wandb:
        # Disable wandb
        os.environ["WANDB_MODE"] = "disabled"
        print("Weights & Biases logging disabled")
    else:
        print("Weights & Biases logging enabled")
    
    # Train B_lineAgent model using behavior cloning
    train_multiple_models(models_config, common_params, args.approach)