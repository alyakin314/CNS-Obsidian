import os
import torch
import numpy as np
import pandas as pd

from abc import abstractmethod

from cns_obsidian.utils import load_variable_json
from cns_obsidian.utils import save_variable_json
from cns_obsidian.utils import calculate_splits
from cns_obsidian.datasets import BaseMultimodalDataset
from cns_obsidian.datasets import BaseMultimodalDataModule


class BaseJournalDataset(BaseMultimodalDataset):
    pass


class BaseJournalDataModule(BaseMultimodalDataModule):
    """
    A data module for handling data loading and batching for training,
    validation, and testing using image-caption data extracted from any journal
    (CNS, PMCOA, etc.). Configured to use a specific tokenizer and image
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
        Defaults to 0.10.
    test_size : float, optional
        The proportion of the dataset to use for testing.
        Defaults to 0.10.
    max_tokens : int
        Maximum number of tokens for the tokenizer.
        Defaults to 256.
    num_workers : int, optional
        Number of subprocesses to use for data loading.
        Defaults to 0.
    distributed : bool, optional
        Whether to setup DistriubtedSamplers and pass them to DataLoaders.
        Defaults to False.
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

    def __init__(
        self, root_dir, processor, dataset_file="dataset.json", *args, **kwargs
    ):
        self.paper_level_split = kwargs.pop("paper_level_split", False)
        super().__init__(root_dir, processor, *args, **kwargs)
        self.dataset_file = dataset_file

    def setup(self, stage=None):
        """
        Prepares the dataset by loading and splitting it into training, validation,
        and test sets.

        Parameters
        ----------
        stage : str, optional
            Stage to set up, either "fit", "validate", "test", or "predict".
            Defaults to None.
        """
        if self.paper_level_split:
            if self.paper_level_split == "exists":
                base_filename = self.dataset_file.rsplit(".", 1)[0]
                extension = self.dataset_file.rsplit(".", 1)[1]
                filenames = [
                    f"{base_filename}_train.{extension}",
                    f"{base_filename}_val.{extension}",
                    f"{base_filename}_test.{extension}",
                ]
            else:
                filenames = self.split_by_paper(
                    self.root_dir,
                    self.dataset_file,
                    self.val_size,
                    self.test_size,
                )
            splits = []
            for filename in filenames:
                dataset = self.DatasetClass(
                    root_dir=self.root_dir,
                    dataset_file=filename,
                    data_key=self.data_key,
                )
                splits.append(dataset)
        else:
            dataset = self.DatasetClass(
                root_dir=self.root_dir,
                dataset_file=self.dataset_file,
                data_key=self.data_key,
            )
            # Calculate the splits
            train_num, val_num, test_num, remainder_num = calculate_splits(
                len(dataset),
                1 - self.val_size - self.test_size,
                self.val_size,
                self.test_size,
                self.batch_size,
            )
            # manually split into train/remainder then remainder into val/test
            splits = torch.utils.data.random_split(
                dataset,
                [train_num, val_num, test_num, remainder_num],
                generator=torch.Generator().manual_seed(314),
            )
        # assign dataset folds to corresponding attributes
        self.train_dataset = splits[0]
        self.val_dataset = splits[1]
        self.test_dataset = splits[2]

        if self.distributed:
            self.setup_distributed(stage=stage)

    @classmethod
    def split_by_paper(cls, root_dir, dataset_file, val_size, test_size):
        """
        Splits a dataset of figures into training, validation, and test
        based on paper handles.

        This function loads data from a JSON or JSONL file, identifies unique
        paper handles, splits the papers into training, validation, and test
        sets, and then saves the split data into separate files.

        **Three new files** will be created with suffixes '_train', '_val', and
        '_test', preserving the original file format. They are overwritten if
        they already exist.

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
        (train_filename, val_filename, test_filename): (str, str, str)
            train_filename: str
                The filename of the training set file.
            val_filename: str
                The filename of the validation set file.
            test_filename: str
                The filename of the test set file.

        Creates
        _______
        Three new files with suffixes '_train', '_val', and '_test' in the same
        format (JSON or JSONL) and place as the {dataset_file} file.
        """
        captions_path = os.path.join(root_dir, dataset_file)
        # Load the JSON data
        data = load_variable_json(captions_path)

        # Convert the list of dictionaries to a pandas DataFrame
        df = pd.DataFrame(data)

        # Extract paper handles
        df["paper_handle"] = cls._grab_paper_names(df)
        # Get unique paper handles
        unique_papers = df["paper_handle"].unique()

        # Calculate the sizes for train, val, and test splits
        train_size = 1 - val_size - test_size

        # Split the paper handles into train, val, test sets
        (
            train_handles,
            val_handles,
            test_handles,
        ) = torch.utils.data.random_split(
            unique_papers.tolist(),
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(314),
        )

        # Convert the handles back to numpy arrays for indexing
        train_handles = np.array(train_handles)
        val_handles = np.array(val_handles)
        test_handles = np.array(test_handles)

        # Split the DataFrame into train, val, and test sets based on paper handles
        train_df = df[df["paper_handle"].isin(train_handles)]
        val_df = df[df["paper_handle"].isin(val_handles)]
        test_df = df[df["paper_handle"].isin(test_handles)]

        # Remove the 'paper_handle' column
        train_df = train_df.drop(columns=["paper_handle"])
        val_df = val_df.drop(columns=["paper_handle"])
        test_df = test_df.drop(columns=["paper_handle"])

        # Generate output filenames without root_dir
        base_filename = dataset_file.rsplit(".", 1)[0]
        extension = dataset_file.rsplit(".", 1)[1]
        train_filename = f"{base_filename}_train.{extension}"
        val_filename = f"{base_filename}_val.{extension}"
        test_filename = f"{base_filename}_test.{extension}"

        # Save the split data into corresponding files with root_dir
        save_variable_json(
            train_df.to_dict(orient="records"),
            os.path.join(root_dir, train_filename),
        )
        save_variable_json(
            val_df.to_dict(orient="records"),
            os.path.join(root_dir, val_filename),
        )
        save_variable_json(
            test_df.to_dict(orient="records"),
            os.path.join(root_dir, test_filename),
        )

        return train_filename, val_filename, test_filename

    @classmethod
    @abstractmethod
    def _grab_paper_names(cls, df):
        pass
