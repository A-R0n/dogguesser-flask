import sys
import logging
 
sys.path.insert(0, '/var/www/dogguesser-flask')
sys.path.insert(0, '/var/www/dogguesser-flask/env/lib/python3.10/site-packages/')
 
# Set up logging
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

activate = '/var/www/dogguesser-flask/env/bin/activate'

with open(activate) as f:
    exec(f_.read(), dict(__file__=activate))

# Import and run the Flask app
from app import app as application