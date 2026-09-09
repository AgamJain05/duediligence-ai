"""
conftest.py  ─  pytest configuration
=====================================
Adds the project root to sys.path so that `src` can be imported
by test files regardless of which directory pytest is run from.
"""
import sys
from pathlib import Path

# Insert project root at the front of the module search path
sys.path.insert(0, str(Path(__file__).resolve().parent))
