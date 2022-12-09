"""Utilities."""

import sys
from typing import Any


def safe_isinstance(obj: Any, class_str: str):
    """Check isinstance without requiring imports.

    Parameters
    ----------
    obj : Any
        The object to check.
    class_str : str
        The class name to check.

    Returns
    -------
    bool
        True if the object is an instance of the class.

    Example
    -------
    >>> model = torch.nn.Module()
    >>> safe_isinstance(model, "torch.nn.Module")
    """

    if not isinstance(class_str, str):
        return False

    module_name, class_name = class_str.rsplit(".", 1)

    if module_name not in sys.modules:
        return False

    module = sys.modules[module_name]
    class_type = getattr(module, class_name, None)

    if class_type is None:
        return False

    return isinstance(obj, class_type)
