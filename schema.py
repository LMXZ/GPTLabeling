from typing import List
from outlines.generate.api import GenerationParameters, SamplingParameters
from outlines.processors import OutlinesLogitsProcessor
from pydantic import BaseModel

from outlines import models
from outlines import generate

from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer
from deepseek_vl.utils.io import load_pil_images
from utils.proxy import RequestUsingProxy
from utils.std import *
import torch
import inspect

class MultiModalTransformers(models.Transformers):
    def __init__(self, deepseek: MultiModalityCausalLM,
                 processor: VLChatProcessor):
        super().__init__(deepseek.language_model, processor.tokenizer)
        self.deepseek = deepseek
        self.processor = processor
    def generate(self, conversation: str | List[str],
                 generation_parameters: GenerationParameters,
                 logits_processor: OutlinesLogitsProcessor | None,
                 sampling_parameters: SamplingParameters) -> str | List[str] | List[List[str]]:
        
        pil_images = load_pil_images(conversation)
        prepare_inputs = self.processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True
        ).to(self.model.device)

        inputs_embeds = self.deepseek.prepare_inputs_embeds(**prepare_inputs)

        inputs = dict(
            input_ids=inputs_embeds,
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.tokenizer.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )
        if (
            "attention_mask"
            not in inspect.signature(self.model.forward).parameters.keys()
        ):
            del inputs["attention_mask"]

        generation_kwargs = self._get_generation_kwargs(
            conversation,
            generation_parameters,
            logits_processor,
            sampling_parameters,
        )
        generated_ids = self._generate_output_seq(conversation, inputs, **generation_kwargs)

        # if single str input and single sample per input, convert to a 1D output
        if isinstance(conversation, str):
            generated_ids = generated_ids.squeeze(0)

        return self._decode_generation(generated_ids)

    def _generate_output_seq(
        self, prompts, inputs, generation_config, **generation_kwargs
    ):
        input_ids = inputs["input_ids"]
        inputs.pop("input_ids")
        output_ids = self.model.generate(
            **inputs, generation_config=generation_config, **generation_kwargs
        )

        # encoder-decoder returns output_ids only, decoder-only returns full seq ids
        if self.model.config.is_encoder_decoder or True:
            generated_ids = output_ids
        else:
            generated_ids = output_ids[:, input_ids.shape[1] :]

        # if batch list inputs AND multiple samples per input, convert generated_id to 3D view
        num_samples = generation_config.num_return_sequences or 1

        if num_samples > 1 and isinstance(prompts, list):
            batch_size = input_ids.size(0)
            num_return_sequences = generation_config.num_return_sequences or 1
            generated_ids = generated_ids.view(batch_size, num_return_sequences, -1)

        return generated_ids
    
with RequestUsingProxy():
    # specify the path to the model
    model_path = "deepseek-ai/deepseek-vl-7b-base"
    vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
    tokenizer = vl_chat_processor.tokenizer

    vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
    vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

model = MultiModalTransformers(vl_gpt, vl_chat_processor)

conversation = [
    {
        "role": "User",
        "content": """Image 1:\n<image_placeholder>
Image 2:\n<image_placeholder>
Image 3:\n<image_placeholder>
Image 4:\n<image_placeholder>
Analyze the feature of the person in the images, in json format.
""",
        "images": [f"./test_data/036300{i}.png" for i in range(1, 5)],
    },
    {"role": "Assistant", "content": ""},
]

schema = json.dumps({
  "title": "person",
  "type": "object",
  "properties": {
    "top_wearing": {"type": "string"},
    "bottom_wearing": {"type": "string"},
    "gender": {"type": "string"},
  },
  "required": ["top_wearing", "bottom_wearing", "gender"]
})

generator = generate.json(model, schema)
result = generator(
    conversation
)
print(result)
# User(name="John", last_name="Doe", id=11)