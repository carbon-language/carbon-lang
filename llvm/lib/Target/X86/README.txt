//===- README.txt - Information about the X86 backend and related files ---===//
//
// This file contains random notes and points of interest about the X86 backend.
//
// Snippets of this document will probably become the final report for CS497
//
//===----------------------------------------------------------------------===//

===========
I. Overview
===========

This directory contains a machine description for the X86 processor.  Currently
this machine description is used for a high performance code generator used by a
LLVM JIT.  One of the main objectives that we would like to support with this
project is to build a nice clean code generator that may be extended in the
future in a variety of ways: new targets, new optimizations, new
transformations, etc.

This document describes the current state of the LLVM JIT, along with
implementation notes, design decisions, and other stuff.


===================================
II. Architecture / Design Decisions
===================================

We designed the infrastructure for the machine specific representation to be as
light-weight as possible, while also being able to support as many targets as
possible with our framework.  This framework should allow us to share many
common machine specific transformations (register allocation, instruction
scheduling, etc...) among all of the backends that may eventually be supported
by the JIT, and unify the JIT and static compiler backends.

At the high-level, LLVM code is translated to a machine specific representation
formed out of MFunction, MBasicBlock, and MInstruction instances (defined in
include/llvm/CodeGen).  This representation is completely target agnostic,
representing instructions in their most abstract form: an opcode, a destination,
and a series of operands.  This representation is designed to support both SSA
representation for machine code, as well as a register allocated, non-SSA form.

Because the M* representation must work regardless of the target machine, it
contains very little semantic information about the program.  To get semantic
information about the program, a layer of Target description datastructures are
used, defined in include/llvm/Target.

Currently the Sparc backend and the X86 backend do not share a common
representation.  This is an intentional decision, and will be rectified in the
future (after the project is done).


=======================
III. Source Code Layout
=======================

The LLVM-JIT is composed of source files primarily in the following locations:

include/llvm/CodeGen
--------------------

This directory contains header files that are used to represent the program in a
machine specific representation.  It currently also contains a bunch of stuff
used by the Sparc backend that we don't want to get mixed up in.

include/llvm/Target
-------------------

This directory contains header files that are used to interpret the machine
specific representation of the program.  This allows us to write generic
transformations that will work on any target that implements the interfaces
defined in this directory.  Again, this also contains a bunch of stuff from the
Sparc Backend that we don't want to deal with.

lib/CodeGen
-----------
This directory will contain all of the target independant transformations (for
example, register allocation) that we write.  These transformations should only
use information exposed through the Target interface, it should not include any
target specific header files.

lib/Target/X86
--------------
This directory contains the machine description for X86 that is required to the
rest of the compiler working.  It contains any code that is truely specific to
the X86 backend, for example the instruction selector and machine code emitter.

tools/jello
-----------
This directory contains the top-level code for the JIT compiler.

test/Regression/Jello
---------------------
This directory contains regression tests for the JIT.  Initially it contains a
bunch of really trivial testcases that we should build up to supporting.


==========================
IV. TODO / Future Projects
==========================

There are a large number of things remaining to do.  Here is a partial list:

Critial path:
-------------

0. Finish providing SSA form.  This involves keeping track of some information
   when instructions are added to the function, but should not affect that API
   for creating new MInstructions or adding them to the program.  There are
   also various FIXMEs in the M* files that need to get taken care of in the
   near term.
1. Finish dumb instruction selector
2. Write dumb register allocator
3. Write assembly language emitter
4. Write machine code emitter

Next Phase:
-----------
1. Implement linear time optimal instruction selector
2. Implement smarter (linear scan?) register allocator

After this project:
-------------------
1. Implement lots of nifty runtime optimizations
2. Implement a static compiler backend for x86
3. Migrate Sparc backend to new representation
4. Implement new spiffy targets: IA64? X86-64? M68k?  Who knows...

Infrastructure Improvements:
----------------------------

1. Bytecode is designed to be able to read particular functions from the
   bytecode without having to read the whole program.  Bytecode reader should be
   extended to allow on demand loading of functions.

2. PassManager needs to be able to run just a single function through a pipeline
   of FunctionPass's.  When this happens, all of our code will become
   FunctionPass's for real.

3. llvmgcc needs to be modified to output 32-bit little endian LLVM files.
   Preferably it will be parameterizable so that multiple binaries need not
   exist.  Until this happens, we will be restricted to using type safe
   programs (most of the Olden suite and many smaller tests), which should be
   sufficient for our 497 project.
