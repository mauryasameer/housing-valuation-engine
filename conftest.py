import os
import sys

# Ensure `from src.*` imports resolve correctly in all test files
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
