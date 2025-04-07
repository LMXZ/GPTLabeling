from utils.std import *
import openai
from openai import OpenAI
from .bases import Labeler
from utils.config import *
import httpx
from utils.decos import NoValidAPIKey, TryAPIKeysUntilSuccess
import base64

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        encoded_bytes = base64.b64encode(img_file.read())
        encoded_str = encoded_bytes.decode("utf-8")
        return encoded_str

def select_api_key(api_keys: List[str]):
    return api_keys[0]

class GPTLabeler():
    def __init__(self, conf=config) -> None:
        self.proxy = conf['proxy']['http']

    @TryAPIKeysUntilSuccess(config['api_groups'][config['selected_api_group']])
    def label(self, file_path: str, description: str, api_key: str=''):
        if api_key == '':
            api_key = self.api_keys[0]
        client = OpenAI(api_key=api_key, http_client=httpx.Client(proxy=self.proxy))
        response = client.responses.create(
            model="gpt-4o",
            input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_to_base64(file_path)}" },
                    {"type": "input_text", "text":
                     f'''"{description}" How closely does the text match the image?
                     Please rate it on a scale of 0 to 10. Only output the score, do not output anything else.''' },
                ],
            }
            ],
        )

        print(response.status)

        print(response.output_text)
