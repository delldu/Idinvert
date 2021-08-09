# python 3.7
"""Utility functions for image editing from latent space."""

import os.path
import numpy as np

import pdb


__all__ = [
    "parse_indices",
    "interpolate",
    "mix_style",
    "get_layerwise_manipulation_strength",
    "manipulate",
    "parse_boundary_list",
]


def parse_indices(obj, min_val=None, max_val=None):
    """Parses indices.

    If the input is a list or tuple, this function has no effect.

    The input can also be a string, which is either a comma separated list of
    numbers 'a, b, c', or a dash separated range 'a - c'. Space in the string will
    be ignored.

    Args:
      obj: The input object to parse indices from.
      min_val: If not `None`, this function will check that all indices are equal
        to or larger than this value. (default: None)
      max_val: If not `None`, this function will check that all indices are equal
        to or smaller than this field. (default: None)

    Returns:
      A list of integers.

    Raises:
      If the input is invalid, i.e., neither a list or tuple, nor a string.
    """
    if obj is None or obj == "":
        indices = []
    elif isinstance(obj, int):
        indices = [obj]
    elif isinstance(obj, (list, tuple, np.ndarray)):
        indices = list(obj)
    elif isinstance(obj, str):
        indices = []
        splits = obj.replace(" ", "").split(",")
        for split in splits:
            numbers = list(map(int, split.split("-")))
            if len(numbers) == 1:
                indices.append(numbers[0])
            elif len(numbers) == 2:
                indices.extend(list(range(numbers[0], numbers[1] + 1)))
    else:
        raise ValueError(f"Invalid type of input: {type(obj)}!")

    assert isinstance(indices, list)
    indices = sorted(list(set(indices)))
    for idx in indices:
        assert isinstance(idx, int)
        if min_val is not None:
            assert idx >= min_val, f"{idx} is smaller than min val `{min_val}`!"
        if max_val is not None:
            assert idx <= max_val, f"{idx} is larger than max val `{max_val}`!"

    return indices


def interpolate(src_codes, dst_codes, step=5):
    """Interpolates two sets of latent codes linearly.

    Args:
      src_codes: Source codes, with shape [num, *code_shape].
      dst_codes: Target codes, with shape [num, *code_shape].
      step: Number of interplolation steps, with source and target included. For
        example, if `step = 5`, three more samples will be inserted. (default: 5)

    Returns:
      Interpolated codes, with shape [num, step, *code_shape].

    Raises:
      ValueError: If the input two sets of latent codes are with different shapes.
    """
    if not (src_codes.ndim >= 2 and src_codes.shape == dst_codes.shape):
        raise ValueError(
            f"Shapes of source codes and target codes should both be "
            f"[num, *code_shape], but {src_codes.shape} and "
            f"{dst_codes.shape} are received!"
        )
    num = src_codes.shape[0]
    code_shape = src_codes.shape[1:]

    a = src_codes[:, np.newaxis]
    b = dst_codes[:, np.newaxis]
    l = np.linspace(0.0, 1.0, step).reshape(
        [step if axis == 1 else 1 for axis in range(a.ndim)]
    )
    results = a + l * (b - a)
    assert results.shape == (num, step, *code_shape)
    pdb.set_trace()

    return results


def mix_style(
    style_codes,
    content_codes,
    num_layers=1,
    mix_layers=None,
    is_style_layerwise=True,
    is_content_layerwise=True,
):
    """Mixes styles from style codes to those of content codes.

    Each style code or content code consists of `num_layers` codes, each of which
    is typically fed into a particular layer of the generator. This function mixes
    styles by partially replacing the codes of `content_codes` from some certain
    layers with those of `style_codes`.

    For example, if both style code and content code are with shape [10, 512],
    meaning to have 10 layers and each employs a 512-dimensional latent code. And
    the 1st, 2nd, and 3rd layers are the target layers to perform style mixing.
    Then the top half of the content code (with shape [3, 512]) will be replaced
    by the top half of the style code (also with shape [3, 512]).

    NOTE: This function also supports taking single-layer latent codes as inputs,
    i.e., setting `is_style_layerwise` or `is_content_layerwise` as False. In this
    case, the corresponding code will be first repeated for `num_layers` before
    performing style mixing.

    Args:
      style_codes: Style codes, with shape [num_styles, *code_shape] or
        [num_styles, num_layers, *code_shape].
      content_codes: Content codes, with shape [num_contents, *code_shape] or
        [num_contents, num_layers, *code_shape].
      num_layers: Total number of layers in the generative model. (default: 1)
      mix_layers: Indices of the layers to perform style mixing. `None` means to
        replace all layers, in which case the content code will be completely
        replaced by style code. (default: None)
      is_style_layerwise: Indicating whether the input `style_codes` are
        layer-wise codes. (default: True)
      is_content_layerwise: Indicating whether the input `content_codes` are
        layer-wise codes. (default: True)
      num_layers

    Returns:
      Codes after style mixing, with shape [num_styles, num_contents, num_layers,
        *code_shape].

    Raises:
      ValueError: If input `content_codes` or `style_codes` is with invalid shape.
    """
    if not is_style_layerwise:
        style_codes = style_codes[:, np.newaxis]
        style_codes = np.tile(
            style_codes,
            [num_layers if axis == 1 else 1 for axis in range(style_codes.ndim)],
        )
    if not is_content_layerwise:
        content_codes = content_codes[:, np.newaxis]
        content_codes = np.tile(
            content_codes,
            [num_layers if axis == 1 else 1 for axis in range(content_codes.ndim)],
        )

    if not (
        style_codes.ndim >= 3
        and style_codes.shape[1] == num_layers
        and style_codes.shape[1:] == content_codes.shape[1:]
    ):
        raise ValueError(
            f"Shapes of style codes and content codes should be "
            f"[num_styles, num_layers, *code_shape] and "
            f"[num_contents, num_layers, *code_shape] respectively, "
            f"but {style_codes.shape} and {content_codes.shape} are "
            f"received!"
        )

    layer_indices = parse_indices(mix_layers, min_val=0, max_val=num_layers - 1)
    if not layer_indices:
        layer_indices = list(range(num_layers))

    num_styles = style_codes.shape[0]
    num_contents = content_codes.shape[0]
    code_shape = content_codes.shape[2:]

    s = style_codes[:, np.newaxis]
    s = np.tile(s, [num_contents if axis == 1 else 1 for axis in range(s.ndim)])
    c = content_codes[np.newaxis]
    c = np.tile(c, [num_styles if axis == 0 else 1 for axis in range(c.ndim)])

    from_style = np.zeros(s.shape, dtype=bool)
    from_style[:, :, layer_indices] = True
    results = np.where(from_style, s, c)
    assert results.shape == (num_styles, num_contents, num_layers, *code_shape)

    return results


def get_layerwise_manipulation_strength(num_layers, truncation_psi, truncation_layers):
    """Gets layer-wise strength for manipulation.

    Recall the truncation trick played on layer [0, truncation_layers):

    w = truncation_psi * w + (1 - truncation_psi) * w_avg

    So, when using the same boundary to manipulate different layers, layer
    [0, truncation_layers) and layer [truncation_layers, num_layers) should use
    different strength to eliminate the effect from the truncation trick. More
    concretely, the strength for layer [0, truncation_layers) is set as
    `truncation_psi`, while that for other layers are set as 1.
    """
    strength = [1.0 for _ in range(num_layers)]
    pdb.set_trace()

    if truncation_layers > 0:
        for layer_idx in range(0, truncation_layers):
            strength[layer_idx] = truncation_psi
    return strength


def manipulate(
    latent_codes,
    boundary,
    start_distance=-5.0,
    end_distance=5.0,
    step=21,
    num_layers=1,
    manipulate_layers=None,
):
    # start_distance = -3.0
    # end_distance = 3.0
    # step = 7
    # num_layers = 14
    # manipulate_layers = [2, 3, 4, 5]

    # boundary.shape -- (1, 14, 512)
    if not (boundary.ndim >= 2 and boundary.shape[0] == 1):
        raise ValueError(
            f"Boundary should be with shape [1, *code_shape] or "
            f"[1, num_layers, *code_shape], but "
            f"{boundary.shape} is received!"
        )

    layer_indices = parse_indices(manipulate_layers, min_val=0, max_val=num_layers - 1)
    if not layer_indices:
        layer_indices = list(range(num_layers))
    assert num_layers > 0

    # layer_indices -- [2, 3, 4, 5]
    x = latent_codes
    # latent_codes.shape -- (20, 14, 512)

    if x.shape[1] != num_layers:
        raise ValueError(
            f"Latent codes should be with shape [num, num_layers, "
            f"*code_shape], where `num_layers` equals to "
            f"{num_layers}, but {x.shape} is received!"
        )

    b = boundary[0]
    # boundary.shape -- (1, 14, 512)
    if b.shape[0] != num_layers:
        raise ValueError(
            f"Boundary should be with shape [num_layers, "
            f"*code_shape], where `num_layers` equals to "
            f"{num_layers}, but {b.shape} is received!"
        )
    # Get layer-wise manipulation strength.
    s = [1.0 for _ in range(num_layers)]
    s = np.array(s).reshape([num_layers if axis == 0 else 1 for axis in range(b.ndim)])
    # pp s.shape -- (14, 1)
    b = b * s

    if x.shape[1:] != b.shape:
        raise ValueError(
            f"Latent code shape {x.shape} and boundary shape " f"{b.shape} mismatch!"
        )
    num = x.shape[0]
    code_shape = x.shape[2:]  # x.shape[2:] -- (512,)

    x = x[:, np.newaxis]
    # x[:, np.newaxis].shape -- (20, 1, 14, 512)

    b = b[np.newaxis, np.newaxis, :]
    # b.shape -- (1, 1, 14, 512)
    l = np.linspace(start_distance, end_distance, step).reshape(
        [step if axis == 1 else 1 for axis in range(x.ndim)]
    )
    # pp l.shape -- (1, 7, 1, 1)
    results = np.tile(x, [step if axis == 1 else 1 for axis in range(x.ndim)])
    # results.shape -- (20, 7, 14, 512)
    is_manipulatable = np.zeros(results.shape, dtype=bool)

    is_manipulatable[:, :, layer_indices] = True
    results = np.where(is_manipulatable, x + l * b, results)
    # (l *b).shape -- (1, 7, 14, 512)

    assert results.shape == (num, step, num_layers, *code_shape)

    return results


def parse_boundary_list(boundary_list_path):
    """Parses boundary list.

    Sometimes, a text file containing a list of boundaries will significantly
    simplify image manipulation with a large amount of boundaries. This function
    is used to parse boundary information from such list file.

    Basically, each item in the list should be with format
    `($NAME, $SPACE_TYPE): $PATH`. `DISABLE` at the beginning of the line can
    disable a particular boundary.

    Sample:

    (age, z): $AGE_BOUNDARY_PATH
    (gender, w): $GENDER_BOUNDARY_PATH
    DISABLE(pose, wp): $POSE_BOUNDARY_PATH

    Args:
      boundary_list_path: Path to the boundary list.

    Returns:
      A dictionary, whose key is a two-element tuple (boundary_name, space_type)
        and value is the corresponding boundary path.

    Raise:
      ValueError: If the given boundary list does not exist.
    """
    if not os.path.isfile(boundary_list_path):
        raise ValueError(f"Boundary list `boundary_list_path` does not exist!")

    pdb.set_trace()

    boundaries = {}
    with open(boundary_list_path, "r") as f:
        for line in f:
            if line[: len("DISABLE")] == "DISABLE":
                continue
            boundary_info, boundary_path = line.strip().split(":")
            boundary_name, space_type = boundary_info.strip()[1:-1].split(",")
            boundary_name = boundary_name.strip()
            space_type = space_type.strip().lower()
            boundary_path = boundary_path.strip()
            boundaries[(boundary_name, space_type)] = boundary_path
    pdb.set_trace()

    return boundaries
