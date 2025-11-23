import os
import json
from huggingface_hub import hf_hub_download


def load_variable_json(file_path):
    """
    Loads data from a JSON or JSONL file.

    This function takes the path to a file that can either be in JSON format
    (a single JSON object or a list of JSON objects) or JSON Lines (JSONL)
    format (each line is a separate JSON object). It appropriately reads and
    loads the data into a list of dictionaries.

    Arguments
    _________
    file_path: str
        Path to the JSON or JSONL file.

    Returns:
    ________
    data: list
        A list of dictionaries containing the parsed JSON objects from the file.

    Raises:
    ValueError:
        If the file extension is not .json or .jsonl.
    """
    # Check the file extension
    _, file_extension = os.path.splitext(file_path)

    # Initialize an empty list to store the data
    data = []

    if file_extension.lower() == ".json":
        # Load the JSON file directly
        with open(file_path, "r") as f:
            data = json.load(f)
    elif file_extension.lower() == ".jsonl":
        # Load the JSONL file line by line
        with open(file_path, "r") as f:
            for line in f:
                data.append(json.loads(line))
    else:
        raise ValueError(
            "Unsupported file format. Please provide a .json or .jsonl file."
        )
    return data


def save_variable_json(data, filename):
    """
    Saves data to a JSON or JSONL file.

    This function takes a list of dictionaries and saves it to a file.
    The file format (JSON or JSONL) is inferred from the file extension.

    Arguments
    _________
    data: list
        A list of dictionaries to be saved.
    filename: str
        The name of the file to save the data to. The extension should be either .json or .jsonl.

    Returns:
    ________
    None
    """
    with open(filename, "w") as f:
        if filename.endswith(".jsonl"):
            for entry in data:
                f.write(json.dumps(entry) + "\n")
        elif filename.endswith(".json"):
            json.dump(data, f, indent=4)
        else:
            raise ValueError(
                "Unsupported file format. Please save as a .json or .jsonl."
            )
    return
