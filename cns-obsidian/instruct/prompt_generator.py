# DISCLAIMER
#
# This file utilizes materials from the Microsoft Research project LLaVA-Med
# (version 1.0.0). Usage of this file is subject to the Microsoft Research
# License Terms.
#
# - This code is used by us for non-commercial, non-revenue generating research
#   purposes only (as allowed per MSR License Terms).
# - Modification of the source code is permitted, but redistribution of the
#   source code, object code, models, or data is not allowed pre MSR License.
# - We include *modified* source code of a *very limited* number of files from
#   LLaVA-Med repo that are specifically required for our project.
#
# For detailed terms, please refer to the full Microsoft Research License Terms
# at: https://github.com/microsoft/LLaVA-Med/tree/v1.0.0
#
# The full source code of the file this file is based off can be found at
# https://github.com/microsoft/LLaVA-Med/blob/v1.0.0/llava/instruct/instruct_generate.py


import os
import json
import numpy as np
import base64


class PromptGenerator:
    """
    Generates prompts that can be used with powerful language models (GPT,
    Claude, and LLaMA) to convert figures and captions into IFT datasets.

    This class configures the model and manages the system prompt, few-shot
    learning examples, and image encoding for prompt generation. The image
    encoding will only be included for the query image-caption pair.

    Parameters
    ----------
    model : str, optional
        The model name to be used. Should contain "gpt", "claude", or "llama".
        Defaults to "gpt-4-turbo".
    few_shots : list, optional
        A list of few-shot examples as dictionaries.
        They should have entries "fig_label", "fig_caption", "in_text_mention",
        and any fields used by the make_an_example callable provided.
        Defaults to an empty list.
    system_prompt : str, optional
        System prompt for the model, will be ignored if the model is LLaMA.
        Defaults to an empty string.
    make_an_example : callable, optional
        Function to create a conversion example of a few shot entry to a good
        esponse conversion.
        Defaults to None.
    use_inline_mentions : bool, optional
        Whether to use inline mentions. If True - expects entries to have a
        field "in_text_mention".
        Defaults to True.
    sample_k : int, optional
        Number of few-shot examples to sample without replacement.
        If None - uses all of them.
        Defaults to None.
    images_root : str, optional
        Root directory for images. If None, images will not be included.
        Defaults to None.
    call_params : dict, optional
        Additional parameters for the API call (e.g., max_tokens, temperature).
        Defaults to an empty dict.
    """

    def __init__(
        self,
        model="gpt-4-turbo",
        few_shots=None,
        system_prompt=None,
        make_an_example=None,
        use_inline_mentions=True,
        sample_k=None,
        images_root=None,
        call_params=None,
    ):
        self.model = model
        self.system_prompt = system_prompt if system_prompt is not None else ""
        self.few_shots = few_shots if few_shots is not None else []
        self.make_an_example = make_an_example
        self.use_inline_mentions = use_inline_mentions
        self.sample_k = sample_k
        self.images_root = images_root
        self.call_params = call_params if call_params is not None else {}

    @staticmethod
    def encode_image(image_path):
        """
        Encodes an image file to a base64 string.

        Parameters
        ----------
        image_path : str
            Path to the image file.

        Returns
        -------
        str
            Base64 encoded string of the image content.
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def few_shot_messages_gen(self, query_context):
        """
        Generates messages for the few-shot examples that will be provided to
        the model. Appends it with a query context that needs to be provided.
        Also handles the system prompt for the GPT-4.

        Parameters
        ----------
        query_context : str
            The context of the query to be included in the messages.

        Returns
        -------
        list
            List of messages formatted for the model.
        """
        messages = []

        if not (self.sample_k is None):
            indices = np.arange(len(self.few_shots))
            np.random.shuffle(indices)
            few_shots_new = [
                self.few_shots[i] for i in indices[: self.sample_k]
            ]
        else:
            few_shots_new = self.few_shots
        for ex in few_shots_new:
            messages += [
                {
                    "role": "user",
                    "content": self.context_gen(ex),
                },
                {
                    "role": "assistant",
                    "content": self.make_an_example(ex),
                },
            ]
        messages.append({"role": "user", "content": query_context})
        return messages

    def context_gen(self, sample, use_image=False):
        """
        Generates context for a given sample.

        This includes figure caption, context, and optionally the image.

        Parameters
        ----------
        sample : dict
            The sample data including figure and context information.
        use_image : bool, optional
            Whether to include the image in the context. Defaults to False.

        Returns
        -------
        str or list
            The generated context string or a list including image data.
        """
        # TODO: To make this general, rather than journal - focused, this
        # should also be handled by a callable provided.
        ctx = []
        if self.use_inline_mentions and sample["in_text_mention"]:
            for sent in sample["in_text_mention"]:
                if isinstance(sent, dict):
                    sent = sent["tokens"]
                ctx.append(sent)
        ret = f"Figure Caption:\n{sample['fig_label']}: {sample['fig_caption']}"
        if len(ctx):
            ret += "\n\nFigure Context:\n\t- {ctx}".format(
                ctx="\n\t- ".join(ctx)
            )
        if use_image and not (self.images_root is None):
            image_path = os.path.join(self.images_root, sample["image"])
            base64_image = self.encode_image(image_path)
            if "claude" in self.model:
                ret = [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": base64_image,
                        },
                    },
                    {"type": "text", "text": ret},
                ]
            elif "gpt-4" in self.model:
                ret = [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        },
                    },
                    {"type": "text", "text": ret},
                ]
        return ret

    def wrap_gen_message(self, sample):
        """
        Wraps the sample with the prompt for a context and adds a few few-shot
        examples in front of it.

        Parameters
        ----------
        sample : dict
            The sample data to be used in generating the messages.

        Returns
        -------
        list
            List of wrapped messages including context and few-shot examples.
        """
        text = self.context_gen(sample, use_image=True)
        messages = self.few_shot_messages_gen(text)
        return messages

    def generate_call_dict(self, sample):
        """
        Generates the API call dictionary for the model.

        Parameters
        ----------
        sample : dict
            The sample data to be used for generating the call dictionary.

        Returns
        -------
        dict
            Dictionary formatted for the API call including messages.
        """
        messages = self.wrap_gen_message(sample)

        if "bedrock" in self.model:
            model_id = self.model.split("bedrock:", 1)[1]
            call_dict = {
                "modelId": model_id,
                "body": json.dumps(
                    {
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": self.call_params.get("max_tokens", 1000),
                        "system": self.system_prompt,
                        "messages": messages,
                    }
                ),
            }
        elif "claude" in self.model:
            call_dict = {
                "model": self.model,
                "system": self.system_prompt,
                "messages": messages,
                **self.call_params,
            }
        elif "gpt" in self.model:
            messages.insert(
                0, {"role": "system", "content": self.system_prompt}
            )
            call_dict = {
                "model": self.model,
                "messages": messages,
                **self.call_params,
            }
        elif "llama" in self.model:
            # TODO Consider prepending the system prompt as first message
            messages_str = json.dumps(messages)
            call_dict = {
                "model": self.model,
                "messages": messages_str,
                **self.call_params,
            }
        else:
            raise NotImplementedError("unknown model")
        return call_dict
