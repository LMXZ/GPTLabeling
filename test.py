from labelers.gpt_labeler import GPTLabeler
from utils.std import *
from utils.config import *
import requests
import datetime

labeler = GPTLabeler()

labeler.label('', '')