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

* In thumb mode, short, byte, and bool preferred alignments are currently set
  to 4 to accommodate ISA restriction (i.e. add sp, #imm, imm must be multiple
  of 4).

//===---------------------------------------------------------------------===//

Potential jumptable improvements:

* If we know function size is less than (1 << 16) * 2 bytes, we can use 16-bit
  jumptable entries (e.g. (L1 - L2) >> 1). Or even smaller entries if the
  function is even smaller. This also applies to ARM.

* Thumb jumptable codegen can improve given some help from the assembler. This
  is what we generate right now:

	.set PCRELV0, (LJTI1_0_0-(LPCRELL0+4))
LPCRELL0:
	mov r1, #PCRELV0
	add r1, pc
	ldr r0, [r0, r1]
	cpy pc, r0 
	.align	2
LJTI1_0_0:
	.long	 LBB1_3
        ...

Note there is another pc relative add that we can take advantage of.
     add r1, pc, #imm_8 * 4

We should be able to generate:

LPCRELL0:
	add r1, LJTI1_0_0
	ldr r0, [r0, r1]
	cpy pc, r0 
	.align	2
LJTI1_0_0:
	.long	 LBB1_3

if the assembler can translate the add to:
       add r1, pc, #((LJTI1_0_0-(LPCRELL0+4))&0xfffffffc)

Note the assembler also does something similar to constpool load:
LPCRELL0:
     ldr r0, LCPI1_0
=>
     ldr r0, pc, #((LCPI1_0-(LPCRELL0+4))&0xfffffffc)
