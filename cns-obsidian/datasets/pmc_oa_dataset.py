import os

from cns_obsidian.utils import load_variable_json
from cns_obsidian.datasets import BaseJournalDataset
from cns_obsidian.datasets import BaseJournalDataModule


class PMCOADataset(BaseJournalDataset):
    """
    A custom dataset class for handling PMC-OA image-caption pairs dataset.
    Each directory represents a journal. It processes JSON files containing
    captions associated with each image.

    Parameters
    ----------
    root_dir : str
        Directory containing all journal subdirectories.
    processor : callable
        A callable processor that processes text and images into a batch.
    dataset_file: str, optional
        File within the directory containing the list of samples (usually json).
        Defaults to "deataset.json"
    data_key: str, optional
        If not None - returns entry[data_key].
        Defaults to None.

    Attributes
    ----------
    samples : list
        List of image-caption pairs loaded from the dataset file.
    """

    def _load_samples(self):
        """
        Loads image-caption pairs from the dataset.json file in each journal

        Returns
        _______
        samples: list
            list of tuples of (image, caption). not transformed and untokenized.
        """
        captions_path = os.path.join(self.root_dir, self.dataset_file)
        dataset = load_variable_json(captions_path)
        samples = []
        for entry in dataset:
            image_file = entry["image"]
            image_path = os.path.join(
                self.root_dir,
                "caption_T060_filtered_top4_sep_v0_subfigures",
                image_file,
            )
            if self.data_key is not None:
                entry = entry[self.data_key]
            samples.append((image_path, entry))
        return samples


class PMCOADataModule(BaseJournalDataModule):
    """
    A data module for handling data loading and batching for training,
    validation, and testing using image-caption data extracted from
    PubMedCentral OpenAcces. Configured to use a specific tokenizer and image
    transformations, and organizes the dataset into train, validation, and test
    splits.

    Parameters
    ----------
    root_dir : str
        The directory containing all journal subdirectories.
    processor : callable
        A callable processor that processes text and images into a batch.
    dataset_file : str, optional
        File within the directory containing the list of samples (usually json).
        Defaults to "pmc_oa.jsonl".
    data_key: str, optional
        If not None - treats entry as a dictionary and returns entry[data_key].
        Defaults to None.
    batch_size : int, optional
        Number of samples in each batch of data.
        Defaults to 32.
    val_size : float, optional
        The proportion of the dataset to use for validation.
        Defaults to 0.15.
    test_size : float, optional
        The proportion of the dataset to use for testing.
        Defaults to 0.10.
    format_string : str, optional
        A string formatted like a format string with variables being keys.
        Defaults to "USER: <image>\nCaption this.\nASSISTANT: {caption}"
    max_tokens : int
        Maximum number of tokens for the tokenizer.
        Defaults to 256.
    num_workers : int, optional
        Number of subprocesses to use for data loading.
        Defaults to 0.
    paper_level_split : bool or str, optional
        If True, creates the split at the paper level and uses it.
        If 'exists', assumes the split files already exist and skips creation.
        If False, splits at the figure level.
        Defaults to False.

    Attributes
    ----------
    train_dataset : Dataset
        The subset of the dataset used for training.
    val_dataset : Dataset
        The subset of the dataset used for validation.
    test_dataset : Dataset
        The subset of the dataset used for testing.
    """

    def __init__(self, root_dir, processor, *args, **kwargs):
        dataset_file = kwargs.pop("dataset_file", "pmc_oa.jsonl")
        # If `dataset_file` is not in `args` or `kwargs`, add it to kwargs
        if len(args) == 0:
            kwargs["dataset_file"] = dataset_file
        super().__init__(root_dir, processor, *args, **kwargs)
        self.DatasetClass = PMCOADataset

    @classmethod
    def _grab_paper_names(cls, df):
        return df["image"].apply(lambda x: x.split("_")[0])
