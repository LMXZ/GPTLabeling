from utils.std import *
import openai
from openai import OpenAI
from .bases import Labeler
from utils.config import *
import httpx
from utils.decos import NoValidAPIKey, TryAPIKeysUntilSuccess
import base64
from utils.shifter import Shifter

def image_to_base64(image_path):
    with open(image_path, "rb") as img_file:
        encoded_bytes = base64.b64encode(img_file.read())
        encoded_str = encoded_bytes.decode("utf-8")
        return encoded_str

def select_api_key(api_keys: List[str]):
    return api_keys[0]

class GPTLabeler(Labeler):
    def __init__(self, conf=config) -> None:
        self.proxy = conf['proxy']['https']
        self.model = conf['model']
        self.api = []
        for k, v in conf['accesses'].items():
            for i in v['api']:
                self.api.append((v['base_url'], i))

    @TryAPIKeysUntilSuccess()
    def label(self, file_path: str, description: str, api_key: str=('', '')):
        client = OpenAI(api_key=api_key[1], http_client=httpx.Client(proxy=self.proxy), base_url=api_key[0])
        texts = '\n'.join(['"' + i + '"' for i in description])
        response = client.chat.completions.create(
            model="gpt-4o",
            temperature=0,
            top_p=0,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_to_base64(file_path)}" }},
                        {"type": "text", "text":
                        f'''{texts} How closely does these texts match the image?
Please rate it on a scale of 0 to 100.
The score is divided into two parts, 70 points for the first part and 30 points for the second part.
The score for the first part is: 70*(the number of features in the text that can be found in the image/the total number of features that appear in the text).
The score for the second part indicates the level of detail of the text description, 30 points for very detailed, 0 points for completely irrelevant.
You must answer by calling the function "answer"!
'''                     
                        },
                    ],
                }
            ],
            functions=[
                {
                    "name": "answer",
                    "description": "analyze the score of each text.",
                    "parameters": {
                        "type": "object",
                        "description": "analyze the score of each text.",
                        "properties": {
                            "analysis": {
                                "type": "array",
                                "description": "analyze of each text",
                                "items": {
                                    "type": "object",
                                    "description": "analyze of one text",
                                    "properties": {
                                        "features_analysis": {
                                            "type": "array",
                                            "description": "break the text into separated features, including gender, hair color hair style, hat, skin color, age, top wearing, bottom wearing, shoes, belongings, etc.",
                                            "items": {
                                                "type": "object",
                                                "description": "analysis of one feature.",
                                                "properties": {
                                                    "description": {
                                                        "type": "string",
                                                        "description": "the description of the feature"
                                                    },
                                                    "presented": {
                                                        "type": "boolean",
                                                        "description": "does this feature match the person in the image?"
                                                    }
                                                }
                                            }
                                        },
                                        "part_1_score": {
                                            "type": "integer",
                                            "enum": list(range(71))
                                        },
                                        "part_2_score": {
                                            "type": "integer",
                                            "enum": list(range(31))
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            ],
            function_call={"name": "answer"}  # 强制调用特定函数
        )

        res0 = json.loads(response.choices[0].message.function_call.arguments)
        res = []
        for i in res0["analysis"]:
            cnt = [0, 0]
            for j in i['features_analysis']:
                cnt[j['presented']] += 1
            res.append(cnt)
        return res, res0
