import re


def pattern_match_fig_label(text):
    """
    Finds matches that look like a figure caption in text.
    """
    # Explanation of the regex:
    # (?is) - Flags:
    #   - (?i) case-insensitive (matches 'fig', 'Fig', etc.)
    #   - (?s) dot matches any character including newline
    # \A - Anchors the match at the start of the string.
    # [\n\r\s]* - Matches 0+ whitespace characters at the start.
    # \b - Word boundary ensures 'fig' starts at the beginning of a word.
    # fig(?:ure)?(?:s)? - Matches 'fig' or 'figure' or 'figs' or 'figures'
    # .? - Optionally matches any character (usually punctuation)
    # \s? - Optionally matches a whitespace character
    # ([^\s]*) - Captures non-whitespace characters, likely the figure number.
    pattern_figure = r"(?is)\A[\n\r\s]*\bfig(?:ure)?(?:s)?.?\s?([^\s]*)"
    matches = re.findall(pattern_figure, text, re.DOTALL)
    return matches


def grab_in_text_mentions(text, label):
    """
    Finds the mentions of the figure in the text and cuts out those paragraphs.
    """
    # Explanation of the regex:
    # (?is) - Flags:
    #   - (?i) case-insensitive (matches 'Fig', 'fig', etc.)
    #   - (?s) dot matches any character including newline
    # (?:.(?<!\n\n))* - Non-capturing group:
    #   - . matches any character
    #   - (?<!\n\n) negative lookbehind, not preceded by double newlines
    # fig(?:ure)?(?:s)? - Matches 'fig' or 'figure' or 'figs' or 'figures'
    # .? - Optionally matches any character (usually punctuation)
    # \s? - Optionally matches a whitespace character
    # {label} - Inserts the figure label to match
    # [^0-9] - Ensures label is followed by a non-digit character
    # (?:(?!\n\n).)* - Non-capturing group:
    #   - (?!\n\n) negative lookahead, stops if double newlines
    #   - . matches any character, capturing till the end of the paragraph
    pattern = (
        f"(?is)(?:.(?<!\n\n))*fig(?:ure)?(?:s)?.?\s?{label}[^0-9](?:(?!\n\n).)*"
    )
    matches = re.findall(pattern, text, re.DOTALL)
    return matches


def calculate_splits(
    dataset_length, train_size, val_size, test_size, batch_size
):
    """
    Calculate the number of samples for training, validation, testing, and
    remainder, rounding to full batches.

    Arguments
    ---------
    dataset_length : int
        The total number of samples in the dataset.
    train_size : float
        The proportion of the dataset to use for training.
    val_size : float
        The proportion of the dataset to use for validation.
    test_size : float
        The proportion of the dataset to use for testing.
    batch_size : int
        Number of samples in each batch of data.

    Returns
    -------
    train_num : int
        Number of training samples.
    val_num : int
        Number of validation samples.
    test_num : int
        Number of testing samples.
    remainder_num : int
        Number of remaining samples after dividing into batches.
    """
    train_num = int(train_size * dataset_length / batch_size) * batch_size
    val_num = int(val_size * dataset_length / batch_size) * batch_size
    test_num = int(test_size * dataset_length / batch_size) * batch_size

    remainder_num = dataset_length - train_num - val_num - test_num
    if extra_batches := (remainder_num // batch_size):
        train_num += extra_batches * batch_size
        remainder_num -= extra_batches * batch_size

    return train_num, val_num, test_num, remainder_num


def count_parameters(model):
    # Parameter Counts
    total_params = sum(p.numel() for p in model.parameters())
    requires_grad_true = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    requires_grad_false = sum(
        p.numel() for p in model.parameters() if not p.requires_grad
    )
    excluded_from_optimizer = sum(
        p.numel()
        for p in model.parameters()
        if getattr(p, "exclude_from_optimizer", False)
    )
    included_in_optimizer = sum(
        p.numel()
        for p in model.parameters()
        if not getattr(p, "exclude_from_optimizer", False)
    )

    print(f"Parameter Statistics:")
    print(f"1. Total parameters: {total_params:,}")
    print(f"2. Parameters with requires_grad=True: {requires_grad_true:,}")
    print(f"3. Parameters with requires_grad=False: {requires_grad_false:,}")
    print(f"4. Parameters included in optimizer: {included_in_optimizer:,}")
    print(f"5. Parameters excluded from optimizer: {excluded_from_optimizer:,}")
