import os
from typing import Dict

def ensure_trial_paths(session_id: str, trial_id: int) -> Dict[str, str]:
    base = os.path.join('data', 'sessions', f'{session_id}_session')
    trial_dir = os.path.join(base, f'trial_{trial_id:03d}')
    meta_dir = os.path.join(base, 'meta')
    os.makedirs(trial_dir, exist_ok=True)
    os.makedirs(meta_dir, exist_ok=True)
    return {
        'base': base,
        'trial_dir': trial_dir,
        'timeline_path': os.path.join(trial_dir, 'timeline.jsonl'),
        'meta_dir': meta_dir,
    }
