import json
import boto3
import time
import openai
import anthropic
import requests

from tqdm import tqdm

from cns_obsidian.utils import load_variable_json
from cns_obsidian.utils import save_variable_json
from cns_obsidian.instruct import mc_system_prompt
from cns_obsidian.instruct import mc_few_shot_examples
from cns_obsidian.instruct import make_a_mc_question
from cns_obsidian.instruct import ift_system_prompt
from cns_obsidian.instruct import ift_few_shot_examples
from cns_obsidian.instruct import make_an_ift_conversation
from cns_obsidian.instruct import ddx_system_prompt
from cns_obsidian.instruct import ddx_few_shot_examples
from cns_obsidian.instruct import make_a_ddx
from cns_obsidian.instruct import PromptGenerator


class APICallProcessor:
    """
    Processes API calls to powerful models (GPT, Claude, and LLaMA) that
    convert figures and captions into IFT datsets.

    Parameters
    ----------
    api_key : str
        The API key for the model service.
    model : str, optional
        The model name to be used. Should contain "gpt", "claude", or "llama".
        Defaults to "gpt-4-turbo".
    mode : str, optional
        Mode of operation. Must be in ["ift", "mc", "ddx", "custom"].
        "ift" - regular instruction fine-tunning (question-answer conversations)
        "mc" - multiple choice questions
        "ddx" - one-lines with a differential diagnosis
        "custom" - expects a system_prompts and few shots to be provided.
        Defaults to "custom".
    system_prompt : str, optional
        System prompt for the model. Ignored unless mode="custom".
        Defaults to None.
    few_shots : list, optional
        Few-shot examples for the model. Ignored unless mode="custom".
        Defaults to an empty list.
    make_an_example : callable, optional
        Function to create an example response. Ignored unless mode="custom".
        Defaults to None.
    sample_k : int, optional
        Number of few-shot examples to sample without replacement.
        If None - uses all of them.
        Defaults to None.
    call_params : dict, optional
        Additional parameters for the API call (e.g., max_tokens, temperature).
        Defaults to an empty dict
    handling_time : float, optional
        Time to wait between calls in seconds.
        Defaults to 0.0.
    output_dict_key : str, optional
        Key for the output in the result dictionary.
        Defaults to "question".
    filtering_function : callable, optional
        Function to filter entries to be processed. Should take an data entry
        and return a boolean.
        Defaults to None.
    """

    def __init__(
        self,
        api_key,
        model="gpt-4-turbo",
        mode="custom",
        system_prompt=None,
        few_shots=None,
        make_an_example=None,
        sample_k=None,
        call_params=None,
        handling_time=0.0,
        output_dict_key="question",
        filtering_function=None,
    ):
        self.api_key = api_key
        self.model = model

        if mode == "mc":
            self.system_prompt = mc_system_prompt
            self.few_shots = mc_few_shot_examples
            self.make_an_example = make_a_mc_question
        elif mode == "ift":
            self.system_prompt = ift_system_prompt
            self.few_shots = ift_few_shot_examples
            self.make_an_example = make_an_ift_conversation
        elif mode in ["ddx"]:
            self.system_prompt = ddx_system_prompt
            self.few_shots = ddx_few_shot_examples
            self.make_an_example = make_a_ddx
        elif mode == "custom":
            self.system_prompt = (
                system_prompt if system_prompt is not None else ""
            )
            self.few_shots = few_shots if few_shots is not None else []
            self.make_an_example = make_an_example
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        self.sample_k = sample_k
        self.call_params = call_params if call_params is not None else {}
        self.handling_time = handling_time
        self.output_dict_key = output_dict_key
        self.filtering_function = (
            filtering_function
            if filtering_function is not None
            else lambda x: True
        )

    def process_json(self, input_path, output_path, images_root=None):
        """
        Processes JSON input data to generate API calls and save outputs.

        This method reads the input JSON file, generates prompts using the
        PromptGenerator, makes API calls to the configured model, and saves
        the responses to the output JSON file.

        Parameters
        ----------
        input_path : str
            Path to the input JSON file.
        output_path : str
            Path to the output JSON file.
        images_root : str, optional
            Root directory for images. Defaults to None.
        """
        prompt_generator = PromptGenerator(
            model=self.model,
            system_prompt=self.system_prompt,
            few_shots=self.few_shots,
            make_an_example=self.make_an_example,
            use_inline_mentions=True,
            sample_k=self.sample_k,
            images_root=images_root,
            call_params=self.call_params,
        )

        if "bedrock" in self.model:
            access_key, secret_key = self.api_key.split(":")

            client = boto3.client(
                "bedrock-runtime",
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name="us-east-1",
            )
        elif "gpt" in self.model:
            client = openai.OpenAI(api_key=self.api_key)
        elif "claude" in self.model:
            client = anthropic.Anthropic(api_key=self.api_key)
        elif "llama" in self.model:
            url = (
                "http://10.189.26.12:30080/model/llama3-70b/v1/chat/completions"
            )
            headers = {
                "apiKey": self.api_key,
                "accept": "application/json",
                "Content-Type": "application/json",
            }

        entries = load_variable_json(input_path)
        processed_entries = []

        for entry in tqdm(entries):
            if not self.filtering_function(entry):
                continue

            call_dict = prompt_generator.generate_call_dict(entry)
            self.temp2 = call_dict
            if "bedrock" in self.model:
                response = client.invoke_model(**call_dict)
                response_body = json.loads(response.get("body").read())
                call_output = response_body["content"][0]["text"]
            elif "gpt" in self.model:
                try:
                    response = client.chat.completions.create(**call_dict)
                except openai.BadRequestError as e:
                    if "unsupported image" in str(e):
                        print(f"Error: {e}. Retrying in 5 minutes...")
                        time.sleep(5 * 60)
                        response = client.chat.completions.create(**call_dict)
                call_output = response.choices[0].message.content
            elif "claude" in self.model:
                response = client.messages.create(**call_dict)
                call_output = response.content[0].text
            elif "llama" in self.model:
                response = requests.post(url, headers=headers, json=call_dict)
                call_output = response.json()["choices"][0]["message"][
                    "content"
                ].split("\n\n", 1)[1]
            else:
                raise ValueError("Unsupported model type")

            entry[self.output_dict_key] = call_output
            processed_entries.append(entry)
            save_variable_json(processed_entries, output_path)
            time.sleep(self.handling_time)

        self.temp = processed_entries
