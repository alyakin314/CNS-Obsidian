import os
from cns_obsidian.instruct import APICallProcessor
from cns_obsidian.instruct.to_filter_a_dataset import create_filtering_function

journal = "Neurosurgery"

nsgy_root = f"/gpfs/data/oermannlab/private_data/TheMedScrolls/FiguresJadenTextract/{journal}/"


processor_gpt4o = APICallProcessor(
    model="claude-3-5-sonnet-20240620",
    api_key="",
    mode="ift",
    sample_k=4,
    call_params={"max_tokens": 1024},
    handling_time=1.1,
)
processor_gpt4o.process_json(
    nsgy_root + "dataset_with_in_text_ift.json",
    nsgy_root + "dataset_ift_claude.json",
    nsgy_root + "images/",
)
