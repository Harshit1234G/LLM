import os
import json


def save_state(state: dict, topic: str) -> None:
    """Save the langgraph state for a given topic."""
    filename = os.path.join('data', f'{sanitize_filename(topic)}.json')

    with open(filename, 'w', encoding= 'utf-8') as f:
        json.dump(
            state, 
            f, 
            indent= 2, 
            ensure_ascii= False
        )


def load_state(topic: str) -> dict | None:
    """Load the langgraph state for a given topic if it exists."""
    filename = os.path.join('data', f'{sanitize_filename(topic)}.json')

    if os.path.exists(filename):
        with open(filename, 'r', encoding= 'utf-8') as f:
            state = json.load(f)

        return state
    
    return None


def sanitize_filename(name: str) -> str:
    """Make sure topic names are safe for filenames."""
    return ''.join(c if c.isalnum() or c in ('-', '_') else '' for c in name).lower()
