import sys
import logging
 
sys.path.insert(0, '/var/www/dogguesser-flask')
sys.path.insert(0, '/var/www/dogguesser-flask/env/lib/python3.10/site-packages/')
 
# Set up logging
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)
 
# Import and run the Flask app
from main import app as application