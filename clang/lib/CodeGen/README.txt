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

It may be worth avoiding creation of alloca's for formal arguments
for the common situation where the argument is never written to or has
its address taken. The idea would be to begin generating code by using
the argument directly and if its address is taken or it is stored to
then generate the alloca and patch up the existing code.

In theory, the same optimization could be a win for block local
variables as long as the declaration dominates all statements in the
block.

//===---------------------------------------------------------------------===//

