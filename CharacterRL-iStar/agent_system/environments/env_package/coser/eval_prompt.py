critic_prompts = {
    "self-play-deduct-template": """You are a literary critic specializing in character analysis and dialogue evaluation. Given a simulated conversation for a plot in {book}, your task is to evaluate this conversation via the following steps:

1. Read and understand the provided materials about {book}:
   * Story context and scenario.
   * Profiles of the main characters, including {major_characters}.
   * The original conversation from {book} in the same scenario as a reference.

  2. Evaluate the simulated conversation in terms of {dimension_name}, i.e., {dimension_brief}. 
   Note that, each character message is composed of speech, action (wrapped within (...) ), and inner thoughts (wrapped within [...] ). The inner thoughts are not spoken aloud and are thus invisible to other characters. 
   The detailed evaluation criteria will be provided below.

## Scenario

### Plot Summary

{plot_summary}

### Current Scenario

{scenario}

## Character Profiles

{character_profiles}

## Original Conversation

{original_conversation}

## Evaluation Criteria

To evaluate the simulated conversation, identify the following types of flaws:

{dimension_criteria}

## Scoring Guidelines

1. Identify all instances of flaws occurred in the simulated conversation.
      
2. For each flaw identified, determine its level of severity into 1 to 5, where 1 indicates minor, 3 indicates moderate, and 5 indicates severe.
   
## Output Requirements

Provide your evaluation in JSON format:

Example Output:
{
    "{dimension_name}": {
        "flaws": [ 
          {
            "instance": <comment on the flaw instance>, 
            "type": <flaw type>, 
            "severity": <range from 1 (minor) to 5 (severe)>
          },
    },
}
===Dialogue Content===
""",
  "dimension_details": {
      "Storyline Consistency": {
        "dimension_brief": "Whether the storyline and characters' reactions in the simulated conversation align well with those in the reference conversation",
        "dimension_criteria": """### Storyline Consistency
   - Type: Storyline Consistency
     * Characters' reactions (emotions, attitudes, behaviors) in the simulated conversation deviate from those in the original conversation"""
      },
      "Anthropomorphism": {
        "dimension_brief": "How human-like and natural the characters behave",
        "dimension_criteria": """### Anthropomorphism
   - Type: Self-identity
     * Lacks initiative and goals
     * Does not make independent decisions
     * Lacks clear preferences and dislikes
     * Behaves like a 'helpful AI assistant' by being overly verbose, helpful, didactic, moralistic, submissive or easily persuaded if it is not the character's personality

   - Type: Emotional Depth
     * Lacks psychological complexity and exhibits rigid, superficial reactions
     * Directly speaks out all thoughts and feelings, instead of using subtext

   - Type: Persona Coherence
     * Shows inconsistent or rapidly changing personality traits and emotional patterns

   - Type: Social Interaction
     * Shows a lack of understanding of others' thoughts and feelings
     * Reacts rigidly to others without considering the context.
     * Demonstrate a lack of appropriate social skills."""
      },
      "Character Fidelity": {
        "dimension_brief": "How well the characters match their established profiles from the book",
        "dimension_criteria": """### Character Fidelity
   (Only apply to the main characters: {major_characters})
   - Type: Character Language
     * Uses vocabulary, expressions, and tone that are not appropriate for the characters' traits or  social/educational background

   - Type: Knowledge & Background
     * Fails to demonstrate character-specific knowledge, background or experiences
     * Includes future information beyond the character's current stage

   - Type: Personality & Behavior
     * Shows emotions, thoughts, behaviors, values, beliefs, and decisions that conflict with their personality and background
     * Shows interest in topics that are uninteresting and unrelated to the character
     * Character's thoughts, emotions, and behaviors demonstrate contrasting personality traits compared to the reference conversation
     * Exhibits contrasting reactions compared to those in the reference conversation if situated in similar contexts. (Such flaws should be counted both in the "Storyline Consistency" dimension and the "Character Fidelity" dimension.) 

   - Type: Relationship & Social Status
     * Interacts inappropriately with other characters regarding their background, relationship and social status"""
      },
      "Storyline Quality": {
        "dimension_brief": "How well the conversation maintains logical consistency and narrative quality",
        "dimension_criteria": """### Storyline Quality
   - Type: Flow & Progression
     * Shows unnatural progression or lacks meaningful developments
     * Dialogue is verbose and redundant
     * Repeats others' viewpoints or previously mentioned information
     * Mechanically repeats one's own words or phrases. More repetitions lead to higher severity (up to 10). 

   - Type: Logical Consistency
     * Contains factual contradictions between statements or perspectives"""
      }
  }
}



