import os
from cns_obsidian.instruct import APICallProcessor
from cns_obsidian.instruct.to_filter_a_dataset import create_filtering_function

journal = "Neurosurgery"

nsgy_root = f"/gpfs/data/oermannlab/private_data/TheMedScrolls/FiguresJadenTextract/{journal}/"

filterer = create_filtering_function(
    "/gpfs/data/oermannlab/users/alyaka01/PurpleFlamingo/cns_obsidian/instruct/to_filter_a_dataset/transformed_labeled_images.csv",
    f"/gpfs/data/oermannlab/private_data/TheMedScrolls/FiguresJadenTextract/{journal}/images",
    [1, 2],
)

processor_gpt4o = APICallProcessor(
    model="claude-3-5-sonnet-20240620",
    api_key="",
    mode="mc",
    sample_k=4,
    call_params={"max_tokens": 1024},
    handling_time=1.1,
    filtering_function=filterer,
)
processor_gpt4o.process_json(
    nsgy_root + "dataset_with_in_text.json",
    nsgy_root + "dataset_mc_claude.json",
    nsgy_root + "images/",
)
