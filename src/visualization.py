import matplotlib.pyplot as plt

def show_class_distribution(
        ax: plt.Axes,
        labels: list[str],
        values: list[int],
        is_train: bool = True,
    ) -> None:
    """
    Display a bar chart of class distribution on the given Axes.

    Args:
        ax (plt.Axes): The matplotlib Axes to plot on.
        labels (List[str]): The class labels.
        values (List[int]): The counts for each class.
        is_train (bool): Whether the distribution is for the training set.

    Returns:
        None
    """
    bars = ax.bar(labels, values)
    ax.set_title(f'{"Training" if is_train else "Validation"} Set Class Distribution')
    ax.set_xlabel('Class')
    ax.set_ylabel('Count')
    ax.tick_params(axis='x', rotation=45)

    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, height, str(height), ha='center', va='bottom')
