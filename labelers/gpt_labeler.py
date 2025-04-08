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

class GPTLabeler(Labeler):
    def __init__(self, conf=config) -> None:
        self.proxy = conf['proxy']['http']

    @TryAPIKeysUntilSuccess(config['api_groups'][config['selected_api_group']])
    def label(self, file_path: str, description: str, api_key: str=''):
        if api_key == '':
            api_key = self.api_keys[0]
        client = OpenAI(api_key=api_key, http_client=httpx.Client(proxy=self.proxy))

        if isinstance(description, str):
            prompt = f'''"{description}" Does the text match the image?
Give the probability 0%~100%.
Only output the probability, do not output anything else.'''
        elif isinstance(description, list):
            prompt = ''
            for i in description:
                prompt += f'"{i}"\n'
            prompt += '''Does these texts match the image?
Give the probability 0%~100%.
Output only their probability, each on a separate line, do not output anything else.'''
        else:
            raise Exception('description must be a string or a list of string.')

        response = client.responses.create(
            model="gpt-4o",
            input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_to_base64(file_path)}" },
                    {"type": "input_text", "text": prompt},
                ],
            }
            ], top_p=0
        )

        if isinstance(description, str):
            return int(response.output_text.replace('%', ''))
        else:
            return list(map(int, response.output_text.replace('%', '').split('\n')))

class GPTCoTLabeler(Labeler):
    def __init__(self, conf=config) -> None:
        self.proxy = conf['proxy']['http']

    @TryAPIKeysUntilSuccess(config['api_groups'][config['selected_api_group']])
    def label(self, file_path: str, description: str, api_key: str=''):
        if api_key == '':
            api_key = self.api_keys[0]
        client = OpenAI(api_key=api_key, http_client=httpx.Client(proxy=self.proxy))
        prompt_user = f'''"{description}" To what extent does this text match the person in the image?
Give their scores (0~10).
The scoring criteria are as follows:
- Part 1: All features in the text appear in the image (7 points).
-- 0~2: Text features almost do not appear in the image.
-- 3~4: Some text features do not appear in the image.
-- 5~7: All text features appear in the image.
- Part 2: The text describes the image as detailed as possible (3 points).
-- 0~1: Almost no match.
-- 2: The text describes most of the features of the image.
-- 3: The text covers almost all the features of the image.'''

        response = client.responses.create(
            model="gpt-4o",
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_to_base64(file_path)}" },
                        {"type": "input_text", "text": prompt_user},
                    ],
                }
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "matching_score_estimating",
                    "description": "the estimating process of the matching score",
                    "schema": {
                        "type": "object",
                        "properties": {
                            "part1_estimation": {
                                "type": "object",
                                "description": "the estimation process of part 1",
                                "properties": {
                                    "text_features": {
                                        "type": "array",
                                        "description": "features present in the text, including gender, hairstyle, hair color, skin color, body shape, age, tops, bottoms, shoes, hats, carrying items, etc.",
                                        "items": {
                                            "type": "object",
                                            "properties": {
                                                "feature_description": {
                                                    "type": "string",
                                                    "description": "the description of one feature in the text"
                                                },
                                                "presented_in_the_image": {
                                                    "type": "boolean",
                                                    "description": "is this feature presented in the image?"
                                                }
                                            },
                                            "required": ["feature_description", "presented_in_the_image"],
                                            "additionalProperties": False
                                        }
                                    },
                                    "score": {
                                        "type": "integer",
                                        "description": "the score of this part",
                                        "enum": list(range(8))
                                    }
                                },
                                "required": ["text_features", "score"],
                                "additionalProperties": False
                            },
                            "part2_score": {
                                "type": "integer",
                                "description": "score of part 2",
                                "enum": list(range(4))
                            },
                            "total_score": {
                                "type": "integer",
                                "description": "the sum of the score of the 2 parts",
                                "enum": list(range(11))
                            }
                        },
                        "required": ["part1_estimation", "part2_score", "total_score"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
            top_p=0
        ).output_text

        response = json.loads(response)

        print(response)

        return int(response['total_score'])
