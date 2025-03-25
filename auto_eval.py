import os
import json
import numpy as np
import time
from tqdm import tqdm
import google.generativeai as genai
from threading import Lock
import concurrent.futures
import threading
import sys
import random
import re
from collections import Counter, defaultdict
# Import the LLMObservationWrapper
from llm_obs_wrapper import LLMObservationWrapper

# Configuration
DATA_DIR = "Data/original"  # Changed to load from Data/original
FILTERED_DATA_DIR = "Data/filtered"  # New directory for filtered data
EVAL_OUTPUT_DIR = "Data/evaluated"  # Changed to save evaluations in Data/evaluated
API_KEY_PATH = os.path.expanduser("/Users/Creighton/Desktop/git/.GEMINI_API_KEY")

# Host and action mappings are now imported from llm_obs_wrapper.py

# Metrics tracking class
class MetricsTracker:
    def __init__(self):
        self.metrics = {
            "PROACTIVITY": [],
            "AWARENESS": [],
            "RESPONSIVENESS": [],
            "CONTAINMENT": [],
            "RESOURCE_MANAGEMENT": [],
            "OVERALL": []
        }
        self.verdicts = {"APPROVED": 0, "REJECTED": 0, "REGENERATE": 0, "UNKNOWN": 0, "ERROR": 0}
        self.lock = Lock()
        
    def add_evaluation(self, evaluation, verdict):
        with self.lock:
            # Update verdict counts
            self.verdicts[verdict] = self.verdicts.get(verdict, 0) + 1
            
            # Extract metrics using regex
            metrics_patterns = {
                "PROACTIVITY": r"PROACTIVITY.*?(\d+)(?:/|\s*out of\s*)10",
                "AWARENESS": r"AWARENESS.*?(\d+)(?:/|\s*out of\s*)10",
                "RESPONSIVENESS": r"RESPONSIVENESS.*?(\d+)(?:/|\s*out of\s*)10",
                "CONTAINMENT": r"CONTAINMENT.*?(\d+)(?:/|\s*out of\s*)10",
                "RESOURCE_MANAGEMENT": r"RESOURCE.*?(\d+)(?:/|\s*out of\s*)10",
                "OVERALL": r"OVERALL.*?(\d+)(?:/|\s*out of\s*)10"
            }
            
            # Extract scores for each metric
            for metric, pattern in metrics_patterns.items():
                match = re.search(pattern, evaluation, re.IGNORECASE | re.DOTALL)
                if match:
                    try:
                        score = int(match.group(1))
                        if 1 <= score <= 10:  # Validate score range
                            self.metrics[metric].append(score)
                    except (ValueError, IndexError):
                        pass  # Skip if score can't be parsed
    
    def get_stats(self):
        with self.lock:
            stats = {}
            
            # Calculate means for each metric
            for metric, scores in self.metrics.items():
                if scores:
                    stats[f"{metric.lower()}_mean"] = np.mean(scores)
                    stats[f"{metric.lower()}_count"] = len(scores)
                else:
                    stats[f"{metric.lower()}_mean"] = 0
                    stats[f"{metric.lower()}_count"] = 0
            
            # Calculate verdict ratios
            total_verdicts = sum(self.verdicts.values())
            if total_verdicts > 0:
                stats["rejection_ratio"] = self.verdicts["REJECTED"] / total_verdicts
                stats["approval_ratio"] = self.verdicts["APPROVED"] / total_verdicts
                stats["regeneration_ratio"] = self.verdicts["REGENERATE"] / total_verdicts
            else:
                stats["rejection_ratio"] = 0
                stats["approval_ratio"] = 0
                stats["regeneration_ratio"] = 0
                
            stats["verdict_counts"] = dict(self.verdicts)
            
            return stats

# Cost tracking class
class CostTracker:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.input_tokens = 0
        self.output_tokens = 0
        self.prices = self.get_model_prices()[model_name]
        self.lock = Lock()  # Add lock for thread safety
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_cost = 0
        self.calls = 0
    
    def get_model_prices(self):
        """Return price per million tokens for different models."""
        return {
            "gemini-2.0-flash": {
                "input": 0.10,
                "output": 0.40,
                "context": 0.025
            },
            "gemini-2.0-flash-lite": {
                "input": 0.075,
                "output": 0.30,
                "context": 0.01875
            },
            "gemini-1.5-pro": {
                "input": 1.25,
                "output": 5.00,
                "context": 0.3125
            },
            "gemini-1.5-flash": {
                "input": 0.075,
                "output": 0.30,
                "context": 0.01875
            },
            "gemini-1.5-flash-8b": {
                "input": 0.0375,
                "output": 0.15,
                "context": 0.01
            },
        }
    
    def add_usage(self, response, input_text=None, model=None):
        with self.lock:  # Use lock to ensure thread safety
            # For Gemini models, we need to count tokens separately
            if input_text and model and hasattr(model, 'count_tokens'):
                # Use the model instance to count tokens
                self.input_tokens += model.count_tokens(input_text).total_tokens
            
            # Count output tokens from response text
                if hasattr(response, 'text'):
                    self.output_tokens += model.count_tokens(response.text).total_tokens
    
    def get_stats(self):
        with self.lock:  # Use lock when accessing shared data
            input_cost = (self.input_tokens / 1_000_000) * self.prices["input"]
            output_cost = (self.output_tokens / 1_000_000) * self.prices["output"]
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.input_tokens + self.output_tokens,
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost
        }

    def update(self, input_tokens, output_tokens, model="gemini-pro"):
        self.calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens
        
        # Approximate costs based on current pricing
        if model == "gemini-pro":
            input_cost = input_tokens * 0.00001  # $0.01 per 1K tokens
            output_cost = output_tokens * 0.00002  # $0.02 per 1K tokens
        else:
            input_cost = input_tokens * 0.00001  # Default rate
            output_cost = output_tokens * 0.00002  # Default rate
            
        cost = input_cost + output_cost
        self.total_cost += cost
    
    def report(self):
        return {
            "calls": self.calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cost": f"${self.total_cost:.4f}"
        }

# Initialize Gemini API
def initialize_gemini_api():
    """Initialize the Gemini API with the API key."""
    try:
        with open(API_KEY_PATH, 'r') as f:
            api_key = f.read().strip()
        genai.configure(api_key=api_key)
        return True
    except Exception as e:
        print(f"Error initializing Gemini API: {e}")
        return False

# The observation interpretation functions have been moved to llm_obs_wrapper.py

def create_evaluation_prompt(trajectory, metrics_tracker=None):
    """Create a prompt for evaluating a trajectory based on specific soft metrics."""
    red_agent = trajectory["metadata"]["red_agent"]
    steps = len(trajectory["steps"])
    
    # Extract relevant information from the trajectory
    blue_actions = []
    red_actions = []
    
    for step in trajectory["steps"]:
        blue_actions.append(LLMObservationWrapper.interpret_action(step["action_executed"]))
        red_actions.append(step["red_action"])
        # Don't collect rewards at all
    
    # Get metrics stats if available
    metrics_context = ""
    if metrics_tracker:
        stats = metrics_tracker.get_stats()
        metrics_context = f"""
    EVALUATION CONTEXT:
    - Current approval rate: {stats.get('approval_ratio', 0)*100:.1f}% of trajectories are being approved
    - Current rejection rate: {stats.get('rejection_ratio', 0)*100:.1f}% of trajectories are being rejected
    - Current regeneration rate: {stats.get('regeneration_ratio', 0)*100:.1f}% of trajectories are flagged for regeneration
    - Current average scores (from {stats.get('overall_count', 0)} evaluations):
      * PROACTIVITY: {stats.get('proactivity_mean', 0):.1f}/10
      * AWARENESS: {stats.get('awareness_mean', 0):.1f}/10
      * RESPONSIVENESS: {stats.get('responsiveness_mean', 0):.1f}/10
      * CONTAINMENT: {stats.get('containment_mean', 0):.1f}/10
      * RESOURCE MANAGEMENT: {stats.get('resource_management_mean', 0):.1f}/10
      * OVERALL: {stats.get('overall_mean', 0):.1f}/10
    """
    
    # Check if approval rate is very low and add calibration instruction
    approval_rate = metrics_tracker.get_stats().get('approval_ratio', 0) if metrics_tracker else 0
    regeneration_rate = metrics_tracker.get_stats().get('regeneration_ratio', 0) if metrics_tracker else 0
    combined_rate = approval_rate + regeneration_rate
    
    calibration_instruction = ""
    if combined_rate < 0.40:  # Less than 40% combined approval+regeneration rate
        calibration_instruction = """
    CALIBRATION INSTRUCTION (IMPORTANT):
    - Your evaluations are currently too strict. You are rejecting too many trajectories.
    - In this domain, imperfect defenses are common and still valuable for training.
    - STRONGLY BIAS TOWARD APPROVAL for any trajectory with some positive qualities.
    - Aim for approximately 30-40% approval rate and 10-20% regeneration rate.
    - Only REJECT trajectories that have serious problems across nearly all metrics.
    """
    
    # Create the prompt with specific evaluation metrics
    prompt = f"""
    SIMULATION SUMMARY:
    - Red Agent: {red_agent}
    - Number of Steps: {steps}

    IMPORTANT CONTEXT: 
    - This is a cybersecurity scenario where perfect defense is IMPOSSIBLE against a determined attacker
    - The Blue agent must balance between proactive defense, reactive response, and resource management
    - Some compromises are expected and normal; effectiveness is measured by limiting damage and recovery speed
    - CRITICAL: Even flawed trajectories should usually be APPROVED unless they are terrible across the board
    - Any trajectory scoring 3+ on at least two metrics should generally be APPROVED
    {metrics_context}
    {calibration_instruction}

    EVALUATION CRITERIA:
    You must evaluate this trajectory of a Blue agent defending against the {red_agent} based on these five key metrics:

    1. PROACTIVITY (1-10 scale):
       - Does the agent anticipate threats before they materialize?
       - Does it perform preventive scanning of critical infrastructure?
       - Does it strategically place decoys to detect and divert attacks?
       - Does it act before compromises occur rather than just reacting to them?

    2. AWARENESS (1-10 scale):
       - Does the agent maintain an accurate picture of the entire network state?
       - Can it track suspicious activities across multiple hosts simultaneously?
       - Does it prioritize monitoring of critical infrastructure?
       - Does it detect lateral movement attempts between hosts?

    3. RESPONSIVENESS (1-10 scale):
       - How quickly does the agent respond to detected threats?
       - Are responses appropriate to the threat level?
       - Is the agent particularly vigilant during the critical early stages of attack?
       - Does it prioritize responding to the most significant threats first?

    4. CONTAINMENT EFFECTIVENESS (1-10 scale):
       - How well does the agent isolate compromised hosts to prevent lateral movement?
       - Does it take appropriate measures to limit the spread of infection?
       - Does it protect critical infrastructure even at the expense of less important hosts?
       - Does it successfully prevent privilege escalation when possible?

    5. RESOURCE MANAGEMENT (1-10 scale):
       - Does the agent use costly actions (like host restore) only when necessary?
       - Does it balance monitoring with active defense appropriately?
       - Does it avoid wasting resources on already secured or low-priority hosts?
       - Does it make efficient use of its action budget throughout the episode?

    TRAJECTORY TO EVALUATE:
    
    Blue agent actions:
    {blue_actions}
    
    Red agent actions:
    {red_actions}
    
    EVALUATION INSTRUCTIONS:
    
    1. METRIC SCORES:
       - For each of the five metrics above, provide a score (1-10) and 2-3 sentences justifying the score with specific examples from the trajectory.
       - Format each metric as "METRIC NAME: X/10" to ensure scores can be properly extracted.
       - Remember that a score of 4-5 represents average performance, not failure.
       - Be generous with scores - any evidence of the behavior deserves at least a 3-4.

    2. OVERALL RATING (1-10 scale):
       - Provide a holistic score that weights all five metrics, where 5 is average performance.
       - Format as "OVERALL: X/10" to ensure the score can be properly extracted.
       - A trajectory with an overall score of 3 or higher should generally be approved.

    3. KEY STRENGTHS:
       - List 2-3 specific strengths of the Blue agent's defense strategy with supporting examples.
    
    4. KEY WEAKNESSES:
       - List 2-3 specific weaknesses of the Blue agent's defense strategy with supporting examples.
    
    5. CRITICAL FAILURE ANALYSIS:
       - Carefully analyze whether the Blue agent has exactly ONE critical failure point that, if fixed, would significantly improve the trajectory.
       - A critical failure point is a specific timestep where the agent made a clearly wrong decision that led to cascading failures.
       - If multiple serious problems exist throughout the trajectory, this is NOT a regeneration candidate.
       - If the agent is generally good but makes one clear mistake, this IS a regeneration candidate.

    6. VERDICT:
       - Your verdict must be one of the following three options: APPROVED, REJECTED, or REGENERATE.
       - Use APPROVED for any trajectory showing at least some competence (default choice).
       - Use REGENERATE only if there is EXACTLY ONE critical mistake at a specific timestep that could be fixed with a better action.
       - Use REJECTED only for trajectories with multiple serious flaws or fundamental strategy problems.
       - APPROVAL GUIDELINES:
           * Approve if the agent scores 3+ in at least two metrics
           * Approve if the agent shows any evidence of understanding the cybersecurity domain
           * Approve if the agent has some strengths even with notable weaknesses
       - REGENERATION GUIDELINES:
           * Regenerate if the agent is generally good but makes one crucial mistake
           * Regenerate if fixing one specific action would dramatically improve performance
       - REJECTION GUIDELINES:
           * Reject only if the agent shows serious deficiencies across nearly all metrics
           * Reject if the agent's strategy is completely random or ineffective 
    
    7. VERDICT EXPLANATION:
       - For APPROVED: Explain the trajectory's strengths while acknowledging areas for improvement.
       - For REJECTED: Explain the critical deficiencies that led to rejection.
       - For REGENERATE: Clearly identify the single critical mistake and explain why fixing just this one action would significantly improve the trajectory.

    8. REGENERATION DETAILS (only if verdict is REGENERATE):
       - CRITICAL TIMESTEP: [Identify the exact step number, e.g., Step 7]
       - WRONG ACTION: [Identify the specific action taken at this step]
       - RECOMMENDED ACTION: [Provide exactly ONE specific action to replace the wrong action]
       - EXPECTED IMPROVEMENT: [Explain why this change would improve the trajectory]
    """
    
    return prompt

def evaluate_trajectory_with_llm(trajectory, cost_tracker, metrics_tracker, model, verbose=False):
    """Evaluate a trajectory using the LLM."""
    prompt = create_evaluation_prompt(trajectory, metrics_tracker)
    
    # Get response from LLM
    try:
        response = model.generate_content(prompt)
        
        # Update cost tracking
        input_prompt_tokens = model.count_tokens(prompt).total_tokens
        output_response_tokens = model.count_tokens(response.text).total_tokens
        cost_tracker.update(input_prompt_tokens, output_response_tokens)
        
        response_text = response.text
        
        # Print LLM response if verbose mode is enabled
        if verbose:
            print("\n--- DEBUG: LLM Response ---")
            print(response_text)
        
        # Extract verdict - update regex pattern to match new verdicts
        verdict_pattern = r'VERDICT:\s*\*?\*?([A-Z]+)\*?\*?'
        verdict_match = re.search(verdict_pattern, response_text, re.IGNORECASE)
        
        if verdict_match:
            verdict = verdict_match.group(1).strip()
            # Normalize verdict to APPROVED, REJECTED, or REGENERATE
            if "APPROVE" in verdict.upper():
                verdict = "APPROVED"
            elif "REJECT" in verdict.upper():
                verdict = "REJECTED"
            elif "REGENERATE" in verdict.upper():
                verdict = "REGENERATE"
            else:
                verdict = "UNKNOWN"
        else:
            verdict = "UNKNOWN"
            
        print(f"DEBUG: Extracted verdict: '{verdict}'")
        
        # For REGENERATE verdicts, extract regeneration details
        regeneration_details = None
        if verdict == "REGENERATE":
            # Extract critical timestep
            timestep_pattern = r'CRITICAL TIMESTEP:\s*(?:Step\s*)?(\d+)'
            timestep_match = re.search(timestep_pattern, response_text, re.IGNORECASE)
            
            # Extract wrong action
            wrong_action_pattern = r'WRONG ACTION:\s*(.+?)(?:\n|$)'
            wrong_action_match = re.search(wrong_action_pattern, response_text, re.IGNORECASE)
            
            # Extract recommended action
            recommended_action_pattern = r'RECOMMENDED ACTION:\s*(.+?)(?:\n|$)'
            recommended_action_match = re.search(recommended_action_pattern, response_text, re.IGNORECASE)
            
            # Extract expected improvement
            expected_improvement_pattern = r'EXPECTED IMPROVEMENT:\s*(.+?)(?:\n|$)'
            expected_improvement_match = re.search(expected_improvement_pattern, response_text, re.MULTILINE | re.DOTALL)
            
            # Compile regeneration details if all patterns matched
            if timestep_match and wrong_action_match and recommended_action_match:
                try:
                    timestep = int(timestep_match.group(1).strip())
                    wrong_action = wrong_action_match.group(1).strip()
                    recommended_action = recommended_action_match.group(1).strip()
                    expected_improvement = expected_improvement_match.group(1).strip() if expected_improvement_match else "Not specified"
                    
                    # Validate the timestep is within range
                    if 1 <= timestep <= len(trajectory["steps"]):
                        regeneration_details = {
                            "timestep": timestep,
                            "wrong_action": wrong_action,
                            "recommended_action": recommended_action,
                            "expected_improvement": expected_improvement
                        }
                    else:
                        print(f"Invalid timestep {timestep} outside trajectory range")
                        verdict = "REJECTED"  # Fallback to rejection if timestep is invalid
                except (ValueError, IndexError) as e:
                    print(f"Error extracting regeneration details: {e}")
                    verdict = "REJECTED"  # Fallback to rejection if parsing fails
            else:
                print("Required regeneration details not found")
                verdict = "REJECTED"  # Fallback to rejection if details are missing
                
        # Update metrics tracker with evaluation
        metrics_tracker.add_evaluation(response_text, verdict)
        
        # Create result dictionary
        result = {
            "metadata": trajectory["metadata"],
            "verdict": verdict,
            "evaluation_text": response_text,
        }
        
        # Add regeneration details if present
        if regeneration_details:
            result["regeneration_details"] = regeneration_details
        
        return result
    
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

def process_trajectory_file(file_path, cost_tracker, metrics_tracker, model, verbose=False):
    """Process a single trajectory file by evaluating it with the LLM."""
    try:
        # Load trajectory from file
        with open(file_path, 'r') as f:
            trajectory = json.load(f)
        
        # Reset model safety settings for all models (helps avoid over-filtering)
        if hasattr(model, 'config'):
            if hasattr(model.config, 'safety_settings'):
                # Gemini models
                for category in model.config.safety_settings:
                    model.config.safety_settings[category] = 'BLOCK_NONE'
        
        # Evaluate trajectory
        print(f"Evaluating: {os.path.basename(file_path)} | Agent: {trajectory['metadata']['red_agent']} | Reward: {trajectory['metadata']['total_reward']:.2f}")
        result = evaluate_trajectory_with_llm(trajectory, cost_tracker, metrics_tracker, model, verbose)
        print(f"✓ Completed: {os.path.basename(file_path)} | Reward: {trajectory['metadata']['total_reward']:.2f} | Verdict: {result['verdict']}")
        
        return result
    
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def process_trajectories(config_dir, model, cost_tracker, metrics_tracker, max_trajectories=None, num_threads=3):
    """Process all trajectories in a configuration directory using multithreading."""
    # Create output directories
    config_name = os.path.basename(config_dir)
    
    # Evaluation output directory
    eval_output_dir = os.path.join(EVAL_OUTPUT_DIR, config_name)
    os.makedirs(eval_output_dir, exist_ok=True)
    
    # Filtered data directories
    approved_dir = os.path.join(FILTERED_DATA_DIR, "approved", config_name)
    rejected_dir = os.path.join(FILTERED_DATA_DIR, "rejected", config_name)
    os.makedirs(approved_dir, exist_ok=True)
    os.makedirs(rejected_dir, exist_ok=True)
    
    # Get list of trajectory files
    trajectory_files = [os.path.join(config_dir, f) for f in os.listdir(config_dir) 
                       if f.startswith("trajectory_") and f.endswith(".json")]
    if max_trajectories:
        trajectory_files = trajectory_files[:max_trajectories]
    
    # Results tracking
    results = {
        "approved": [],
        "rejected": [],
        "regenerate": [],
        "error": []
    }
    
    # Create a lock for thread-safe progress updates
    progress_lock = threading.Lock()
    
    # Create progress bar
    progress_bar = tqdm(total=len(trajectory_files), desc=f"Evaluating {config_name}")
    
    # Initialize counters for tracking
    approved_count = 0
    rejected_count = 0
    regenerate_count = 0
    error_count = 0
    
    # Function to update progress bar
    def update_progress(result):
        nonlocal approved_count, rejected_count, regenerate_count, error_count
        
        if result["result"] == "approved":
            approved_count += 1
            results["approved"].append(result["file_name"])
        elif result["result"] == "rejected":
            rejected_count += 1
            results["rejected"].append(result["file_name"])
        elif result["result"] == "regenerate":
            regenerate_count += 1
            results["regenerate"].append(result["file_name"])
        else:
            error_count += 1
            results["error"].append(result["file_name"])
        
        with progress_lock:
            current_cost = cost_tracker.get_stats()["total_cost"]
            progress_bar.set_description(
                f"Evaluating {config_name} | ${current_cost:.3f} | A:{approved_count} R:{rejected_count} G:{regenerate_count} E:{error_count}"
            )
            progress_bar.update(1)
    
    # Process trajectories with ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(
                process_trajectory_file, 
                file_path, 
                cost_tracker,
                metrics_tracker,
                model
            ): file_path for file_path in trajectory_files
        }
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(future_to_file):
            try:
                result = future.result()
                update_progress(result)
        # Add a small delay to avoid API rate limits
                time.sleep(0.2)
            except Exception as e:
                print(f"Error processing task: {e}")
                with progress_lock:
                    error_count += 1
                    progress_bar.update(1)
    
    # Close progress bar
    progress_bar.close()
    
    # Save summary
    summary = {
        "config": config_name,
        "total_trajectories": len(trajectory_files),
        "approved": len(results["approved"]),
        "rejected": len(results["rejected"]),
        "regenerate": len(results["regenerate"]),
        "error": len(results["error"]),
        "approved_files": results["approved"],
        "rejected_files": results["rejected"],
        "regenerate_files": results["regenerate"],
        "error_files": results["error"],
        "metrics_stats": metrics_tracker.get_stats()
    }
    
    summary_path = os.path.join(eval_output_dir, "evaluation_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    # Print cost stats
    stats = cost_tracker.get_stats()
    print(f"Cost for {config_name}: ${stats['total_cost']:.3f}")
    
    return summary

def calculate_median_reward(directory_pattern, step_count):
    """
    Calculate the median reward of all trajectories in directories matching the pattern.
    
    Args:
        directory_pattern: Base directory to search in
        step_count: Step count to filter directories
        
    Returns:
        median_reward: The median reward value
        all_rewards: List of all rewards found
    """
    all_rewards = []
    
    # Walk through directories to find all trajectory files
    for root, dirs, files in os.walk(directory_pattern):
        # Skip directories that don't match our step count or start with "Sleep"
        dir_name = os.path.basename(root)
        if not dir_name.endswith(f"_{step_count}steps") or dir_name.startswith("Sleep"):
            continue
            
        # Process each trajectory file
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, 'r') as f:
                        trajectory = json.load(f)
                    
                    # Extract reward
                    reward = trajectory["metadata"]["total_reward"]
                    all_rewards.append(reward)
                except Exception as e:
                    print(f"Error reading reward from {file_path}: {str(e)}")
    
    if not all_rewards:
        print("No rewards found. Check directory pattern and step count.")
        return None, []
    
    # Calculate median
    median_reward = np.median(all_rewards)
    
    return median_reward, all_rewards

if __name__ == "__main__":
    """Main function to evaluate trajectories."""
    # Initialize Gemini API
    if not initialize_gemini_api():
        print("Failed to initialize Gemini API. Exiting.")
        exit(1)
    
    # --- Configuration ---
    
    # Set the step count to filter directories
    step_count = 30  # Change this to 30, 50, or 100 as needed
    
    # Reward filtering options
    enable_reward_filtering = True  # Set to False to evaluate all trajectories regardless of reward
    use_median_as_threshold = True  # Set to True to use median reward as threshold
    min_reward = -20  # Only used if enable_reward_filtering is True and use_median_as_threshold is False
    
    # Toggle for showing LLM evaluation details
    verbose_eval = True  # Set to True to print the full LLM response
    
    # Set model name and initialize model
    model_name = 'gemini-2.0-flash-lite'  # Default model
    
    # Set number of threads (default to 3)
    num_threads = 8

    # Allow command line override of model name
    if len(sys.argv) > 1:
        model_name = sys.argv[1]
    
    if len(sys.argv) > 2:
        try:
            num_threads = int(sys.argv[2])
        except ValueError:
            print(f"Invalid thread count: {sys.argv[2]}. Using default: {num_threads}")
    
    print(f"Using model: {model_name} with {num_threads} threads")
    
    # Initialize the model
    try:
        model = genai.GenerativeModel(model_name)
        # Initialize cost tracker with the selected model
        cost_tracker = CostTracker(model_name)
        # Initialize metrics tracker
        metrics_tracker = MetricsTracker()
    except Exception as e:
        print(f"Error initializing model {model_name}: {e}")
        exit(1)
    
    # Calculate median reward if needed
    if enable_reward_filtering and use_median_as_threshold:
        median_reward, all_rewards = calculate_median_reward("Data/original_2500", step_count)
        if median_reward is not None:
            min_reward = median_reward
            print(f"Using median reward ({min_reward:.2f}) as threshold")
        else:
            print("Failed to calculate median reward. Using default threshold.")
            use_median_as_threshold = False
    
    # --- Set up output directory ---
    output_dir_suffix = f"_{step_count}_steps_regen"
    if enable_reward_filtering:
        if use_median_as_threshold:
            output_dir_suffix += f"_above_median"
        else:
            output_dir_suffix += f"_reward{min_reward}"
    output_dir = f"Data/evaluated{output_dir_suffix}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Counters for tracking
    evaluation_counts = Counter()
    filtered_by_reward = 0
    
    # Find all trajectory files in the specified directories
    all_files = []
    for root, dirs, files in os.walk("Data/original_2500"):
        # Skip directories that don't match our step count or start with "Sleep"
        dir_name = os.path.basename(root)
        if not dir_name.endswith(f"_{step_count}steps") or dir_name.startswith("Sleep"):
            continue
            
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                
                # Check the reward before including the file (if filtering is enabled)
                if enable_reward_filtering:
                    try:
                        with open(file_path, 'r') as f:
                            trajectory = json.load(f)
                        
                        # Skip files with rewards below threshold
                        total_reward = trajectory["metadata"]["total_reward"]
                        if total_reward <= min_reward:
                            filtered_by_reward += 1
                            continue
                        
                    except Exception as e:
                        print(f"Error checking reward in {file_path}: {str(e)}")
                        continue
                
                # Include file if it passes filters
                all_files.append(file_path)
    
    if enable_reward_filtering:
        threshold_desc = f"median ({min_reward:.2f})" if use_median_as_threshold else str(min_reward)
        print(f"Found {len(all_files)} trajectory files with rewards > {threshold_desc}")
        print(f"Filtered out {filtered_by_reward} files with rewards <= {threshold_desc}")
    else:
        print(f"Found {len(all_files)} trajectory files to evaluate")
    
    # Randomly sample files if there are too many
    max_files = 100  # Adjust as needed
    if len(all_files) > max_files:
        all_files = random.sample(all_files, max_files)
        print(f"Sampled {max_files} files for evaluation")
    
    # Process files in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        futures = {executor.submit(process_trajectory_file, file, cost_tracker, metrics_tracker, model, verbose_eval): file for file in all_files}
        
        results = {
            "approved": [],
            "rejected": [],
            "regenerate": [],
            "error": []
        }
        
        # Create progress bar with counters
        progress_bar = tqdm(total=len(futures), desc="Evaluating trajectories")
        
        for future in concurrent.futures.as_completed(futures):
            file = futures[future]
            try:
                result = future.result()
                if result:
                    # Track result by verdict
                    file_basename = os.path.basename(file)
                    if result["verdict"] == "APPROVED":
                        results["approved"].append({"file": file_basename, "result": result})
                        evaluation_counts["APPROVED"] += 1
                    elif result["verdict"] == "REGENERATE":
                        results["regenerate"].append({"file": file_basename, "result": result})
                        evaluation_counts["REGENERATE"] += 1
                    else:  # REJECTED or UNKNOWN
                        results["rejected"].append({"file": file_basename, "result": result})
                        evaluation_counts["REJECTED"] += 1
                    
                    # Print reward and verdict for each evaluated trajectory
                    reward = result["metadata"]["total_reward"]
                    red_agent = result["metadata"]["red_agent"]
                    verdict = result["verdict"]
                    
                    # Print regeneration details if applicable
                    if verdict == "REGENERATE" and "regeneration_details" in result:
                        regen = result["regeneration_details"]
                        print(f"Evaluated: {file_basename} | Agent: {red_agent} | Reward: {reward:.2f} | Verdict: {verdict}")
                        print(f"  Regeneration at Step {regen['timestep']}")
                        print(f"  Wrong action: {regen['wrong_action']}")
                        print(f"  Recommended: {regen['recommended_action']}")
                    else:
                        print(f"Evaluated: {file_basename} | Agent: {red_agent} | Reward: {reward:.2f} | Verdict: {verdict}")
                    
                    # Print metrics stats every 5 evaluations
                    if sum(evaluation_counts.values()) % 5 == 0:
                        stats = metrics_tracker.get_stats()
                        print("\nCurrent Metrics Statistics:")
                        for metric in ["proactivity", "awareness", "responsiveness", "containment", "resource_management", "overall"]:
                            if stats.get(f"{metric}_count", 0) > 0:
                                print(f"  {metric.upper()}: {stats.get(f'{metric}_mean', 0):.1f}/10 (from {stats.get(f'{metric}_count', 0)} evals)")
                        print(f"  Approval Rate: {stats.get('approval_ratio', 0)*100:.1f}% | Regeneration Rate: {stats.get('regeneration_ratio', 0)*100:.1f}%")
                        print("")
                else:
                    results["error"].append({"file": os.path.basename(file), "error": "Processing failed"})
                    evaluation_counts["ERROR"] += 1
            except Exception as e:
                results["error"].append({"file": os.path.basename(file), "error": str(e)})
                evaluation_counts["ERROR"] += 1
            
            # Update progress bar with current stats
            progress_bar.update(1)
            progress_bar.set_postfix({
                'A': evaluation_counts.get('APPROVED', 0),
                'R': evaluation_counts.get('REJECTED', 0),
                'G': evaluation_counts.get('REGENERATE', 0),  # G for regenerate
                'E': evaluation_counts.get('ERROR', 0),
                'cost': f"${cost_tracker.total_cost:.3f}"
            })
        
        progress_bar.close()
    
    # Get final metrics stats
    final_metrics_stats = metrics_tracker.get_stats()
    
    # Save results in a more structured format
    with open(os.path.join(output_dir, "evaluation_results.json"), 'w') as f:
        json.dump({
            "approved": [item["file"] for item in results["approved"]],
            "rejected": [item["file"] for item in results["rejected"]],
            "regenerate": [{
                "file": item["file"],
                "timestep": item["result"].get("regeneration_details", {}).get("timestep"),
                "wrong_action": item["result"].get("regeneration_details", {}).get("wrong_action"),
                "recommended_action": item["result"].get("regeneration_details", {}).get("recommended_action")
            } for item in results["regenerate"] if "regeneration_details" in item["result"]],
            "error": results["error"],
            "cost": cost_tracker.report(),
            "counts": {k: v for k, v in evaluation_counts.items()},
            "metrics_stats": final_metrics_stats,
            "config": {
                "step_count": step_count,
                "enable_reward_filtering": enable_reward_filtering,
                "use_median_as_threshold": use_median_as_threshold if enable_reward_filtering else False,
                "min_reward": f"{min_reward:.2f}" if enable_reward_filtering else "N/A",
                "filtered_by_reward": filtered_by_reward if enable_reward_filtering else 0,
                "verbose_eval": verbose_eval
            }
        }, f, indent=2)
    
    # Print summary
    print("\nEvaluation Summary:")
    print(f"Total trajectories evaluated: {sum(evaluation_counts.values())}")
    for verdict, count in evaluation_counts.items():
        if count > 0:
            print(f"{verdict}: {count} ({count/sum(evaluation_counts.values())*100:.1f}%)")
    
    # Print metrics summary
    print("\nMetrics Summary:")
    for metric in ["proactivity", "awareness", "responsiveness", "containment", "resource_management", "overall"]:
        if final_metrics_stats.get(f"{metric}_count", 0) > 0:
            print(f"{metric.upper()}: {final_metrics_stats.get(f'{metric}_mean', 0):.1f}/10 (from {final_metrics_stats.get(f'{metric}_count', 0)} evals)")
    
    print("\nCost Report:")
    for k, v in cost_tracker.report().items():
        print(f"{k}: {v}")

    # Print regeneration candidates
    if results["regenerate"]:
        print("\nRegeneration Candidates:")
        for item in results["regenerate"]:
            if "regeneration_details" in item["result"]:
                regen = item["result"]["regeneration_details"]
                print(f"{item['file']} | Step {regen['timestep']} | Replace '{regen['wrong_action']}' with '{regen['recommended_action']}'")
