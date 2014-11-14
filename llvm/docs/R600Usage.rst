============================
User Guide for R600 Back-end
============================

Introduction
============

The R600 back-end provides ISA code generation for AMD GPUs, starting with
the R600 family up until the current Sea Islands (GCN Gen 2).


Assembler
=========

The assembler is currently a work in progress and not yet complete.  Below
are the currently supported features.

SOPP Instructions
-----------------

Unless otherwise mentioned, all SOPP instructions that with an operand
accept a integer operand(s) only.  No verification is performed on the
operands, so it is up to the programmer to be familiar with the range
or acceptable values.

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

