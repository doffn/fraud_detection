def ip_to_int(ip_address):
    """
    Converts a dotted-decimal IP address string to an integer.
    Returns None if conversion fails.
    """
    try:
        parts = list(map(int, ip_address.split('.')))
        return parts[0] * 256**3 + parts[1] * 256**2 + parts[2] * 256 + parts[3]
    except (AttributeError, ValueError):
        return None
