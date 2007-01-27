//===---------------------------------------------------------------------===//
// Random ideas for the ARM backend (Thumb specific).
//===---------------------------------------------------------------------===//

* Add support for compiling functions in both ARM and Thumb mode, then taking
  the smallest.
* Add support for compiling individual basic blocks in thumb mode, when in a 
  larger ARM function.  This can be used for presumed cold code, like paths
  to abort (failure path of asserts), EH handling code, etc.

* Thumb doesn't have normal pre/post increment addressing modes, but you can
  load/store 32-bit integers with pre/postinc by using load/store multiple
  instrs with a single register.

* Make better use of high registers r8, r10, r11, r12 (ip). Some variants of add
  and cmp instructions can use high registers. Also, we can use them as
  temporaries to spill values into.

* If we know function size is less than (1 << 16) * 2 bytes, we can use 16-bit
  jumptable entries (e.g. (L1 - L2) >> 1). Or even smaller entries if the
  function is even smaller. This also applies to ARM.
