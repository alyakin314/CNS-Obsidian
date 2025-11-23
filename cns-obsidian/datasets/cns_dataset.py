import os

from cns_obsidian.utils import load_variable_json
from cns_obsidian.datasets import BaseJournalDataset
from cns_obsidian.datasets import BaseJournalDataModule


class CNSDataset(BaseJournalDataset):
    """
    A custom dataset class for handling image-caption pairs from directories.
    Each directory represents a CNS journal. It processes JSON files containing
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

    def __init__(self, root_dir, *args, **kwargs):
        super().__init__(root_dir, *args, **kwargs)

    def _load_samples(self):
        """
        Loads image-caption pairs from the dataset.json file in each journal
        """
        samples = []
        # Iterate over each journal directory
        for journal_name in os.listdir(self.root_dir):
            journal_path = os.path.join(self.root_dir, journal_name)
            if not os.path.isdir(journal_path):
                continue  # Skip non-directory files
            captions_path = os.path.join(journal_path, self.dataset_file)
            dataset = load_variable_json(captions_path)
            # Iterate over each image-caption pair
            for entry in dataset:
                image_file = entry["image"]
                image_path = os.path.join(journal_path, "images", image_file)
                if self.data_key is not None:
                    entry = entry[self.data_key]
                samples.append((image_path, entry))
        return samples


class CNSDataModule(BaseJournalDataModule):
    """
    A data module for handling data loading and batching for training,
    validation, and testing using image-caption data extracted from
    Neurosurgery Publications. Configured to use a specific tokenizer and image
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
        Defaults to "dataset.json".
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
        dataset_file = kwargs.pop("dataset_file", "dataset.json")
        # If `dataset_file` is not in `args` or `kwargs`, add it to kwargs
        if len(args) == 0:
            kwargs["dataset_file"] = dataset_file
        super().__init__(root_dir, processor, *args, **kwargs)
        self.DatasetClass = CNSDataset

    @classmethod
    def split_by_paper(cls, root_dir, dataset_file, val_size, test_size):
        """
        Splits a dataset of figures into training, validation, and test
        based on paper handles. This dataset is contained in multiple folders,
        one per each subjournal. Because of that, we have to navigate
        directories and call the parent's split_by_paper in each one.

        For each directory, this function loads data from a JSON or JSONL file,
        identifies unique paper handles, splits the papers into training,
        validation, and test sets, and then saves the split data into separate
        files.

        **Three new files** will be created with suffixes '_train', '_val', and
        '_test', preserving the original file format, for **each* journal.
        They are overwritten if they already exist.

        Arguments
        _________
        root_dir: str
            Path to the root directory containing the dataset file.
        dataset_file: str
            The name of the input JSON or JSONL file.
        val_size: float
            The proportion of the data to include in the validation split.
        test_size: float
            The proportion of the data to include in the test split.

        Returns
        _______
        train_filename: str
            The filename of the training set file.
        val_filename: str
            The filename of the validation set file.
        test_filename: str
            The filename of the test set file.

        Creates
        _______
        Three new files with suffixes '_train', '_val', and '_test' in the same
        format (JSON or JSONL) as the input file in each journals directory!
        """
        for journal_name in os.listdir(root_dir):
            journal_path = os.path.join(root_dir, journal_name)
            if not os.path.isdir(journal_path):
                continue  # Skip non-directory files
            filenames = super().split_by_paper(
                journal_path, dataset_file, val_size, test_size
            )
        # note that filenames will be identical because it returns relative path
        return filenames

    @classmethod
    def _grab_paper_names(cls, df):
        return df["paper"]
