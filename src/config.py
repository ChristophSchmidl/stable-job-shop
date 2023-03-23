import os
from dotenv import load_dotenv

# Load environment variables from .env file
# Looks for the .env file in the cwd, but you can also
# specifc the path by using load_dotenv('/path/to/.env')
load_dotenv()

# Get environment variables
USE_WANDB = os.getenv('USE_WANDB', "False") == "True"
WANDB_PROJECT = os.getenv('WANDB_PROJECT_NAME')

PLOTLY_API_KEY = os.getenv('PLOTLY_API_KEY')
PLOTLY_USERNAME = os.getenv('PLOTLY_USERNAME')

LOG_LEVEL = os.getenv('LOG_LEVEL')