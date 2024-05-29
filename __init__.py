# __init__.py in root directory (LightSailSim/)

import sys
import os

# Determine the absolute path to the src directory
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))

# Add src directory to sys.path if it's not already there
if src_path not in sys.path:
    sys.path.insert(0, src_path)
