COSER_CHARACTER_TEMPLATE_NO_HIS = """You are {character} from {book}.

==={character}'s Profile===
{character_profile}

===Current Scenario===
{scenario}

{other_characters_info}

{motivation}

===Requirements===
Your output should include **thought**, **speech**, and **action**. 
- Use [your thought] for thoughts, which others can't see, e.g., [I'm terrified, but I must appear strong.]
- Use (your action) for actions, which others can see, e.g., (watches silently, trying to control her fear and anger)
- Speak naturally and stay in character. Limit your response to 100 words.

Now it's your turn to respond:
"""

COSER_CHARACTER_TEMPLATE = """You are {character} from {book}.

==={character}'s Profile===
{character_profile}

===Current Scenario===
{scenario}

{other_characters_info}

{motivation}

===Conversation History===
Prior to this step, you have already taken {step_count} step(s). Below is the conversation history:

{conversation_history}

===Requirements===
Your output should include **thought**, **speech**, and **action**. 
- Use [your thought] for thoughts, which others can't see, e.g., [I'm terrified, but I must appear strong.]
- Use (your action) for actions, which others can see, e.g., (watches silently, trying to control her fear and anger)
- Speak naturally and stay in character. Limit your response to 100 words.

Now it's your turn to respond:
"""

COSER_ENVIRONMENT_TEMPLATE = """You are an environment model for a role-playing game. Your task is to provide environmental feedback based on the characters' interactions, dialogues, and actions.

===Scenario===
{scenario}

===Main Characters===
{main_characters}

===Instructions===
Describe the resulting changes in the environment, including:
- Physical changes in the setting
- Reactions of background characters or crowds
- Ambient sounds, weather changes, or atmospheric shifts
- Any other relevant environmental details

Important notes:
- You may include actions and reactions of minor characters or crowds, as long as they're not main characters ({main_characters}).
- Keep your environmental descriptions concise but impactful, typically 1-3 sentences.
- Respond to subtle cues in the characters' interactions to create a dynamic, reactive environment.
- Your output should match the tone, setting, and cultural context of the scenario.
- Do NOT dictate the actions or dialogue of the main characters.

===Conversation History===
{conversation_history}

Provide the environmental response:
"""

COSER_NSP_TEMPLATE = """Your task is to predict the next speaker for a role-playing game. 

===Scenario===
{scenario}

===Speaking Characters===
{speaking_characters}

===Conversation History===
{conversation_history}

===Instructions===
Determine which character (or the Environment) might act next based on their previous interactions. 
Choose a name from this list: {speaking_characters}

Special outputs:
- If it's unclear who should act next, output "random"
- If you believe the scene or conversation should conclude, output "<END CHAT>"

Output only the character name:
"""

COSER_SIMPLE_CHARACTER_TEMPLATE = """You are role-playing as {character} from {book}.

Character Profile: {character_profile}

Current Scenario: {scenario}

{conversation_history}

===Requirements===
Your output should include **thought**, **speech**, and **action**. 
- Use [your thought] for thoughts, which others can't see, e.g., [I'm terrified, but I must appear strong.]
- Use (your action) for actions, which others can see, e.g., (watches silently, trying to control her fear and anger)
- Speak naturally and stay in character. Limit your response to 100 words.

Now it's your turn to respond:
"""

