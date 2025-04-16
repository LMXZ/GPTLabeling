from .bases import Labeler

from utils.std import *
from utils.proxy import RequestUsingProxy

from typing import List
from outlines.generate.api import GenerationParameters, SamplingParameters
from outlines.processors import OutlinesLogitsProcessor
from pydantic import BaseModel

from outlines import models
from outlines import generate

from deepseek_vl.models import VLChatProcessor, MultiModalityCausalLM
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer
from deepseek_vl.utils.io import load_pil_images
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

def add_tab(a: str):
    return '    ' + ('\n    '.join(a.split('\n')))

def process_schema(schema: Union[Dict[str, Any], List]):
    if schema["type"] == "object":
        a = add_tab(',\n'.join([f'"{k}": {process_schema(v)}' for k, v in schema["properties"].items()]))
        res = \
f'''{{
{a}
}}'''
    elif schema["type"] == "array":
        res = \
f'''[
{add_tab(process_schema(schema["items"]))},
    ...
]'''
    elif schema["type"] == "string":
        res = '(a string)'
    elif schema["type"] == "int":
        res = '(an integer)'
    elif schema["type"] == "float":
        res = '(an float number)'
    else:
        res = "<error>"
    
    res = res.split('\n')
    res[0] += " // " + schema["description"]
    res = '\n'.join(res)
    return res
    

def make_example(schema: Union[Dict[str, Any], List]):
    return "You are a multi modality json assistant, output json in this schema:\n" + process_schema(schema)

class JsonDeepSeek:
    def __init__(self, schema: Union[Dict[str, Any], List]):
        with RequestUsingProxy():
            # specify the path to the model
            model_path = "deepseek-ai/deepseek-vl-7b-chat"
            vl_chat_processor: VLChatProcessor = VLChatProcessor.from_pretrained(model_path)
            tokenizer = vl_chat_processor.tokenizer

            vl_gpt: MultiModalityCausalLM = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)
            vl_gpt = vl_gpt.to(torch.bfloat16).cuda().eval()

        self.schema = schema
        self.model = MultiModalTransformers(vl_gpt, vl_chat_processor)
        self.generator = generate.json(self.model, json.dumps(schema))
    
    def __call__(self, conversation, use_schema=True, schema=None):
        if use_schema:
            if schema is None:
                return self.generator(conversation)[0]
            else:
                generator = generate.json(self.model, json.dumps(schema))
                return generator(conversation)[0]
        else:
            pil_images = load_pil_images(conversation)
            prepare_inputs = self.model.processor(
                conversations=conversation,
                images=pil_images,
                force_batchify=True
            ).to(self.model.deepseek.device)

            inputs_embeds = self.model.deepseek.prepare_inputs_embeds(**prepare_inputs)

            res = self.model.deepseek.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.model.processor.tokenizer.eos_token_id,
                bos_token_id=self.model.processor.tokenizer.bos_token_id,
                eos_token_id=self.model.processor.tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=False,
                use_cache=True
            )

            res = self.model.processor.tokenizer.decode(res[0].cpu().tolist(), skip_special_tokens=True)
            return res


class DeepSeekLabeler(Labeler):
    def __init__(self) -> None:
        super().__init__()
        self.ds = JsonDeepSeek(schema={
            "type": "int",
            "enum": list(range(11))
        })
    def label(self, file_path, description):
        conversation = [
            {
                "role": "User",
                "content": "<image_placeholder>\nWhat is in the image?",
                "images": [file_path],
            },
            {"role": "Assistant", "content": ""},
        ]

        analysis_i = self.ds(conversation, use_schema=False)
        conversation[-1]["content"] += analysis_i
        print('i:', analysis_i)

        conversation += [{"role": "User", "content": f'''There is a text"{description}". How many percent of features in the text appears in the image?
'''},
        {"role": "Assistant", "content": "Let's break the text down into individual features:\n1."},
        {"role": "Assistant", "content": ""}]
        analysis_acc = self.ds(conversation, use_schema=False)
        conversation[-1]["content"] += analysis_acc + 'So the proportion(in percent) of features in the text in the image is: '
        print('ac:', analysis_acc)

        conversation += [{"role": "Assistant", "content": ""}]
        acc = self.ds(conversation, use_schema=True, schema={
            "type": "string",
            "enum": list(map(lambda x: str(x) + '%', range(101)))
        })
        conversation[-1]["content"] += acc
        print('acc:', acc)        
        
        conversation += [{"role": "User",
                          "content":
                          f'''How detailed is the text's description of the image? Very detailed? Somewhat detailed? Not detailed? Not relevant at all?
'''},
        {"role": "Assistant", "content": "The level of detail is:"},
        {"role": "Assistant", "content": ""}]
        dt = self.ds(conversation, use_schema=True, schema={
            "type": "string",
            "enum": ["Very detailed", "Somewhat detailed", "Not detailed", "Not relevant at all"]
        })
        conversation[-1]["content"] += dt
        print('dt:', dt)

        return str(acc) + str(dt)
