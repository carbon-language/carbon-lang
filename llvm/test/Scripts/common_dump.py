def dataToHex(d):
    """ Convert the raw data in 'd' to an hex string with a space every 4 bytes.
    """
    bytes = []
    for i,c in enumerate(d):
        byte = ord(c)
        hex_byte = hex(byte)[2:]
        if byte <= 0xf:
            hex_byte = '0' + hex_byte
        if i % 4 == 3:
            hex_byte += ' '
        bytes.append(hex_byte)
    return ''.join(bytes).strip()
