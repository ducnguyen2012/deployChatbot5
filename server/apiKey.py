from dotenv import load_dotenv, dotenv_values
import os
load_dotenv()

def api():
    '''
    return api key if you adjust this!
    '''
    return os.getenv("MY_SECRET_KEY")