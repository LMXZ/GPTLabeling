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
        content = []
        for i, fp in enumerate(file_path):
            content += [
                {"type": "input_text", "text": f"image{i}:\n"},
                {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_to_base64(fp)}" },
                {"type": "input_text", "text": "\n"},
            ]
        for i in description:
            content.append({"type": "input_text", "text": f'"{i}"\n'})
        response = client.responses.create(
            model="gpt-4o",
            input=[
                {
                    "role": "user",
                    "content": content + [
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
                            "text_analysis": {
                                "type": "array",
                                "description": "text analysis of all texts",
                                "items": {
                                    "type": "object",
                                    "description": "text analysis of a text",
                                    "properties": {
                                        "feature_analysis": {
                                            "type": "array",
                                            "description": "features presented in the text, including gender, hairstyle, hair color, skin color, body shape, age, tops, bottoms, shoes, hats, carrying items, etc. Each list item contains only ONE feature above.",
                                            "items": {
                                                "type": "object",
                                                "description": "feature analysis of a feature item of the text",
                                                "properties": {
                                                    "feature_desciption": {
                                                        "type": "string",
                                                        "description": "the description of one feature in the text"
                                                    },
                                                    "presentation": {
                                                        "type": "array",
                                                        "description": "indicates if this text feature item presented in each image",
                                                        "items": {
                                                            "type": "boolean",
                                                            "description": "indicates if this text feature item presented in an image"
                                                        }
                                                    }
                                                },
                                                "required": ["feature_desciption", "presentation"],
                                                "additionalProperties": False
                                            }
                                        },
                                        "score": {
                                            "type": "array",
                                            "description": "The matching score between the text and each image",
                                            "items": {
                                                "type": "object",
                                                "description": "The matching score between the text and an image",
                                                "properties": {
                                                    "part1_score": {
                                                        "type": "integer",
                                                        "description": "The part1 matching score between the text and an image",
                                                        "enum": list(range(8))
                                                    },
                                                    "part2_score": {
                                                        "type": "integer",
                                                        "description": "The part2 matching score between the text and an image",
                                                        "enum": list(range(4))
                                                    },
                                                    "total_score": {
                                                        "type": "integer",
                                                        "description": "The sum of part1 and part2",
                                                        "enum": list(range(11))
                                                    }
                                                },
                                                "required": ["part1_score", "part2_score", "total_score"],
                                                "additionalProperties": False,
                                            }
                                        }
                                    },
                                    "required": ["feature_analysis", "score"],
                                    "additionalProperties": False,
                                }
                            }
                        },
                        "required": ["text_analysis"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            },
            top_p=0
        ).output_text

        response = json.loads(response)

        json.dump(response, open('log.json', 'w'))

        res = []

        for i in response["text_analysis"]:
            res.append([j["total_score"] for j in i["score"]])

        return res
