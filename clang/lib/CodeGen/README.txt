IRgen optimization opportunities.

//===---------------------------------------------------------------------===//

The common pattern of
--
short x; // or char, etc
(x == 10)
--
generates an zext/sext of x which can easily be avoided.

//===---------------------------------------------------------------------===//

Bitfields accesses can be shifted to simplify masking and sign
extension. For example, if the bitfield width is 8 and it is
appropriately aligned then is is a lot shorter to just load the char
directly.

//===---------------------------------------------------------------------===//

Bitfields should not reload the stored value just to return the
correct result.

//===---------------------------------------------------------------------===//
