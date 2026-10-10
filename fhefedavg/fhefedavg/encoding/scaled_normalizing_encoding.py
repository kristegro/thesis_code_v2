

def encode(val, scale, bound):
    """Encodes a real number val as an integer using a scaling factor scale and a bound on val,
    by normalizing, scaling and rounding.

    Args:
        val (float): The real to encode.
        scale (int): The scaling factor to use.
        bound (int): The bound on val, |val| <= bound.

    Returns:
        int: The encoded value.
    """
    return round(scale * (val + bound) / (2*bound))

def decode(val, scale, bound):
    """Decodes an integer encoding back to a real number.

    Args:
        val (int): Encoding to decode.
        scale (int): Scaling factor used during encoding.
        bound (int): Bound on the encoded real used during encoding.

    Returns:
        float: The recovered real.
    """
    return (2 * bound * val) / scale - bound