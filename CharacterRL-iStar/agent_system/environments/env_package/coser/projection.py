import re
from typing import List, Tuple

from agent_system.environments.env_package.coser.utils import \
    remove_inner_thoughts


def coser_projection(actions: List[str]) -> Tuple[List[str], List[int]]:
    """
    Project LLM actions into valid CoSER dialogue actions.
    
    This function processes raw LLM outputs to extract valid dialogue responses
    for the CoSER role-playing environment.
    
    Extraction logic:
        1. Remove any character name prefix (e.g., "Harry Potter: ")
        2. Remove inner thoughts marked with [...]
        3. Clean up extra whitespace
    
    Validity logic:
        1. Action should not be empty after processing
    
    Args:
        actions: List of raw LLM output strings
        
    Returns:
        results: List of processed action strings
        valids: List of validity flags (1 for valid, 0 for invalid)
    """
    results: List[str] = []
    valids: List[int] = [1] * len(actions)
    
    for i, action in enumerate(actions):
        processed_action = action.strip()
        
        # Remove common character name prefixes
        # Pattern: "Character Name: dialogue"
        name_prefix_pattern = r'^[A-Z][a-zA-Z\s]+:\s*'
        processed_action = re.sub(name_prefix_pattern, '', processed_action)
        
        # Remove inner thoughts
        processed_action = remove_inner_thoughts(processed_action)
        
        # Check if empty after processing
        if not processed_action:
            results.append("")
            valids[i] = 0
        else:
            results.append(processed_action)
    
    return results, valids


def _extract_speaker_response(text: str, speaker: str) -> str:
    """
    Extract a specific speaker's response from multi-speaker text.
    
    This is useful when the LLM generates multiple characters' responses
    but we only want one specific character's dialogue.
    
    Args:
        text: Raw text possibly containing multiple speakers
        speaker: Name of the speaker to extract
        
    Returns:
        Extracted dialogue for the specified speaker
    """
    lines = text.split('\n')
    current_speaker = None
    current_dialogue = []
    all_dialogues = {}
    
    for line in lines:
        # Check if line starts with a speaker name
        if ':' in line:
            parts = line.split(':', 1)
            potential_speaker = parts[0].strip()
            
            # If this looks like a speaker name
            if potential_speaker and len(potential_speaker.split()) <= 4:
                # Save previous speaker's dialogue
                if current_speaker and current_dialogue:
                    all_dialogues[current_speaker] = '\n'.join(current_dialogue)
                
                # Start new speaker
                current_speaker = potential_speaker
                current_dialogue = [parts[1].strip() if len(parts) > 1 else '']
                continue
        
        # Add to current speaker's dialogue
        if current_speaker:
            current_dialogue.append(line)
    
    # Save last speaker's dialogue
    if current_speaker and current_dialogue:
        all_dialogues[current_speaker] = '\n'.join(current_dialogue)
    
    # Return dialogue for requested speaker
    if speaker in all_dialogues:
        return all_dialogues[speaker]
    
    # If speaker not found, return original text
    return text

