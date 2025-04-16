import torch
from transformers import AutoModelForCausalLM

from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from deepseek_vl.utils.io import load_pil_images
from utils.proxy import RequestUsingProxy
from utils.std import *

from pydantic import BaseModel
from outlines import models, generate

import outlines
from tqdm import tqdm

os.environ["NVIDIA_VISIBLE_DEVICES"] = "3"

with RequestUsingProxy():
    # specify the path to the model
    model_path = "deepseek-ai/deepseek-vl-7b-chat"
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

## single image conversation example
data = json.load(open('test_data/reid_raw.json'))[-4:]

## multiple images (or in-context learning) conversation example
# conversation = [
#     {
#         "role": "User",
#         "content": "<image_placeholder>A dog wearing nothing in the foreground, "
#                    "<image_placeholder>a dog wearing a santa hat, "
#                    "<image_placeholder>a dog wearing a wizard outfit, and "
#                    "<image_placeholder>what's the dog wearing?",
#         "images": [
#             "images/dog_a.png",
#             "images/dog_b.png",
#             "images/dog_c.png",
#             "images/dog_d.png",
#         ],
#     },
#     {"role": "Assistant", "content": ""}
# ]
def gen(conversation):
    # load images and prepare for inputs
    pil_images = load_pil_images(conversation)
    prepare_inputs = vl_chat_processor(
        conversations=conversation,
        images=pil_images,
        force_batchify=True
    ).to(vl_gpt.device)

    # run image encoder to get the image embeddings
    inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)

    # run the model to get the response
    outputs = vl_gpt.language_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=prepare_inputs.attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        max_new_tokens=512,
        do_sample=False,
        use_cache=True
    )

    answer = tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    return answer

def score(image, text):
    conversation = [
        {
            "role": "User",
            "content": f"""<image_placeholder>
    "{text}", please score the similarity between the image and the text, scale from 0 to 10.
    """,
            "images": [image],
        },
        {"role": "Assistant", "content": ""},
    ]

    ans = gen(conversation)
    conversation[1]["content"] = ans + "so my final answer is:"
    conversation += [{"role": "Assistant", "content": ""}]
    return int(gen(conversation))

images = []
texts = []

for i in data:
    images += ['test_data/' + i['file_path']]
    texts += i['captions']

ans = []
for ii, i in enumerate(images):
    ans += [[]]
    for jj, j in enumerate(texts):
        s = score(i, j)
        ans[-1] += [s]
        print('--'* 5, ii, ' ', jj, s)

for i in ans:
    print(i)