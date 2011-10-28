"Collection of tools for displaying bit representation of numbers."""

def binary(n, width=None):
    """
    Return a list of (0|1)'s for the binary representation of n where n >= 0.
    If you specify a width, it must be > 0, otherwise it is ignored.  The list
    could be padded with 0 bits if width is specified.
    """
    l = []
    if width and width <= 0:
        width = None
    while n > 0:
        l.append(1 if n&1 else 0)
        n = n >> 1

    if width:
        for i in range(width - len(l)):
            l.append(0)

    l.reverse()
    return l

def twos_complement(n, width):
    """
    Return a list of (0|1)'s for the binary representation of a width-bit two's
    complement numeral system of an integer n which may be negative.
    """
    val = 2**(width-1)
    if n >= 0:
        if n > (val-1):
            return None
        # It is safe to represent n with width-bits.
        return binary(n, width)

    if n < 0:
        if abs(n) > val:
            return None
        # It is safe to represent n (a negative int) with width-bits.
        return binary(val*2 - abs(n))

# print binary(0xABCD)
# [1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 1]
# print binary(0x1F, 8)
# [0, 0, 0, 1, 1, 1, 1, 1]
# print twos_complement(-5, 4)
# [1, 0, 1, 1]
# print twos_complement(7, 4)
# [0, 1, 1, 1]
# print binary(7)
# [1, 1, 1]
# print twos_complement(-5, 64)
# [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1]

