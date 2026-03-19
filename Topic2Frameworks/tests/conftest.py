"""Pytest configuration. Ensures Topic2Frameworks is on path for imports."""
import sys
from pathlib import Path

# Add parent (Topic2Frameworks) to path so task7_multi_llm_history etc. can be imported
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
