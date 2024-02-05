import sys
import logging
 
sys.path.insert(0, '/var/www/dogguesser-flask')
sys.path.insert(0, '/var/www/dogguesser-flask/env/lib/python3.10/site-packages/')
 
# Set up logging
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

activate_this = '/var/www/dogguesser-flask/env/bin/activate_this.py'

with open(activate_this) as f:
    exec(f_.read(), dict(__file__=activate_this))

# Import and run the Flask app
from app import app
application = app()