============================
User Guide for R600 Back-end
============================

Introduction
============

The R600 back-end provides ISA code generation for AMD GPUs, starting with
the R600 family up until the current Volcanic Islands (GCN Gen 3).


Assembler
=========

The assembler is currently considered experimental.

For syntax examples look in test/MC/R600.

Below some of the currently supported features (modulo bugs).  These
all apply to the Southern Islands ISA, Sea Islands and Volcanic Islands
are also supported but may be missing some instructions and have more bugs:

DS Instructions
---------------
All DS instructions are supported.

MUBUF Instructions
------------------
All non-atomic MUBUF instructions are supported.

SMRD Instructions
-----------------
Only the s_load_dword* SMRD instructions are supported.

SOP1 Instructions
-----------------
All SOP1 instructions are supported.

SOP2 Instructions
-----------------
All SOP2 instructions are supported.

SOPC Instructions
-----------------
All SOPC instructions are supported.

SOPP Instructions
-----------------

Unless otherwise mentioned, all SOPP instructions that have one or more
operands accept integer operands only.  No verification is performed
on the operands, so it is up to the programmer to be familiar with the
range or acceptable values.

s_waitcnt
^^^^^^^^^

s_waitcnt accepts named arguments to specify which memory counter(s) to
wait for.

.. code-block:: nasm

   // Wait for all counters to be 0
   s_waitcnt 0

   // Equivalent to s_waitcnt 0.  Counter names can also be delimited by
   // '&' or ','.
   s_waitcnt vmcnt(0) expcnt(0) lgkcmt(0)

   // Wait for vmcnt counter to be 1.
   s_waitcnt vmcnt(1)

VOP1, VOP2, VOP3, VOPC Instructions
-----------------------------------

All 32-bit and 64-bit encodings should work.

The assembler will automatically detect which encoding size to use for
VOP1, VOP2, and VOPC instructions based on the operands.  If you want to force
a specific encoding size, you can add an _e32 (for 32-bit encoding) or
_e64 (for 64-bit encoding) suffix to the instruction.  Most, but not all
instructions support an explicit suffix.  These are all valid assembly
strings:

.. code-block:: nasm

   v_mul_i32_i24 v1, v2, v3
   v_mul_i32_i24_e32 v1, v2, v3
   v_mul_i32_i24_e64 v1, v2, v3
