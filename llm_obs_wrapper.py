import numpy as np

# Host and action mappings
HOST_MAPPING = {
    0: "Defender",
    1: "User0",
    2: "User1", 
    3: "User2",
    4: "User3",
    5: "User4",
    6: "Enterprise0",
    7: "Enterprise1",
    8: "Enterprise2",
    9: "OpServer0",
    10: "OPServer1",
    11: "OPServer2",
    12: "UNKNOWN1",
    13: "UNKNOWN2"
}

ACTION_CATEGORIES = {
    "Sleep": [0],
    "Monitor": [1],
    "Analyse": list(range(2, 16)),  # Actions 2-15
    "Remove": list(range(15, 29)),  # Actions 15-28
    "Restore": list(range(132, 146))  # Actions 132-145
}

DECOY_MAPPING = {
    1000: "Enterprise0",
    1001: "Enterprise1",
    1002: "Enterprise2",
    1003: "User1",
    1004: "User2",
    1005: "User3",
    1006: "User4",
    1007: "Defender",
    1008: "OpServer0"
}

class LLMObservationWrapper:
    """
    Wrapper for converting numerical CybORG observations and actions to human-readable format
    suitable for passing to Large Language Models.
    """
    
    @staticmethod
    def interpret_activity(activity):
        """Interpret the activity part of an observation."""
        if np.array_equal(activity, [0, 0]):
            return "None"
        elif np.array_equal(activity, [1, 0]):
            return "Scan"
        elif np.array_equal(activity, [1, 1]):
            return "Exploit"
        return "Unknown"

    @staticmethod
    def interpret_compromised(compromised):
        """Interpret the compromised part of an observation."""
        if np.array_equal(compromised, [0, 0]):
            return "No"
        elif np.array_equal(compromised, [1, 0]):
            return "Unknown"
        elif np.array_equal(compromised, [0, 1]):
            return "User"
        elif np.array_equal(compromised, [1, 1]):
            return "Privileged"
        return "Unknown"

    @staticmethod
    def interpret_observation(observation):
        """Convert a numerical observation to a human-readable format."""
        if not isinstance(observation, np.ndarray):
            observation = np.array(observation)
        
        num_hosts = len(observation) // 4
        hosts_info = []
        
        for i in range(num_hosts):
            base_idx = i * 4
            activity = observation[base_idx:base_idx+2]
            compromised = observation[base_idx+2:base_idx+4]
            
            activity_status = LLMObservationWrapper.interpret_activity(activity)
            compromised_status = LLMObservationWrapper.interpret_compromised(compromised)
            
            host_name = HOST_MAPPING.get(i, f"Unknown Host {i}")
            hosts_info.append({
                "host_id": i,
                "host_name": host_name,
                "activity": activity_status,
                "compromised": compromised_status
            })
        
        return hosts_info

    @staticmethod
    def interpret_action(action):
        """Convert a numerical action to a human-readable format."""
        try:
            action = int(action)
        except:
            # If action is already a string (like from trajectory data)
            return str(action)
        
        # Handle decoy actions (special case)
        if action >= 1000:
            return f"Deploy Decoy on {DECOY_MAPPING.get(action, 'Unknown Host')}"
        
        # Handle regular actions
        for category, action_ids in ACTION_CATEGORIES.items():
            if action in action_ids:
                if category in ["Sleep", "Monitor"]:
                    return f"{category}"
                else:
                    host_id = action - (2 if category == "Analyse" else 15 if category == "Remove" else 132)
                    host_name = HOST_MAPPING.get(host_id, f"Unknown Host {host_id}")
                    return f"{category} on {host_name}"
        
        return f"Unknown Action {action}"

    @staticmethod
    def format_trajectory_for_llm(trajectory):
        """Format a trajectory for LLM evaluation."""
        metadata = trajectory["metadata"]
        steps = trajectory["steps"]
        
        # Format metadata
        formatted_metadata = (
            f"Red Agent: {metadata['red_agent']}\n"
            f"Episode Length: {metadata['steps']} steps\n"
            f"Total Reward: {metadata['total_reward']}\n"
        )
        
        # Format steps
        formatted_steps = []
        for step_data in steps:
            step_num = step_data["step"]
            observation = LLMObservationWrapper.interpret_observation(step_data["observation"])
            action = LLMObservationWrapper.interpret_action(step_data["action"])
            action_executed = LLMObservationWrapper.interpret_action(step_data["action_executed"])
            red_action = step_data["red_action"]
            reward = step_data["reward"]
            
            # Format the observation
            obs_text = "Network Status:\n"
            for host in observation:
                obs_text += f"  - {host['host_name']}: Activity={host['activity']}, Compromised={host['compromised']}\n"
            
            step_text = (
                f"Step {step_num}:\n"
                f"{obs_text}\n"
                f"Blue Action: {action}\n"
                f"Blue Action Executed: {action_executed}\n"
                f"Red Action: {red_action}\n"
                f"Reward: {reward}\n"
            )
            formatted_steps.append(step_text)
        
        return formatted_metadata, formatted_steps

    @staticmethod
    def create_nlp_observation(observation):
        """
        Create a natural language description of an observation suitable for an LLM.
        
        Args:
            observation: The numerical observation array
            
        Returns:
            A string with the natural language description
        """
        hosts_info = LLMObservationWrapper.interpret_observation(observation)
        
        # Format as a paragraph
        nlp_obs = "Current network status: "
        status_parts = []
        
        for host in hosts_info:
            status = f"{host['host_name']} shows {host['activity']} activity"
            if host['compromised'] != "No":
                status += f" and is compromised ({host['compromised']} level)"
            else:
                status += " and is not compromised"
            status_parts.append(status)
        
        nlp_obs += "; ".join(status_parts) + "."
        return nlp_obs
