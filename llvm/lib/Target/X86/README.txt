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

We designed the infrastructure into the generic LLVM machine specific
representation, which allows us to support as many targets as possible with our
framework.  This framework should allow us to share many common machine specific
transformations (register allocation, instruction scheduling, etc...) among all
of the backends that may eventually be supported by LLVM, and ensures that the
JIT and static compiler backends are largely shared.

At the high-level, LLVM code is translated to a machine specific representation
formed out of MachineFunction, MachineBasicBlock, and MachineInstr instances
(defined in include/llvm/CodeGen).  This representation is completely target
agnostic, representing instructions in their most abstract form: an opcode, a
destination, and a series of operands.  This representation is designed to
support both SSA representation for machine code, as well as a register
allocated, non-SSA form.

Because the Machine* representation must work regardless of the target machine,
it contains very little semantic information about the program.  To get semantic
information about the program, a layer of Target description datastructures are
used, defined in include/llvm/Target.

Note that there is some amount of complexity that the X86 backend contains due
to the Sparc backend's legacy requirements.  These should eventually fade away
as the project progresses.


SSA Instruction Representation
------------------------------
Target machine instructions are represented as instances of MachineInstr, and
all specific machine instruction types should have an entry in the
InstructionInfo table defined through X86InstrInfo.def.  In the X86 backend,
there are two particularly interesting forms of machine instruction: those that
produce a value (such as add), and those that do not (such as a store).

Instructions that produce a value use Operand #0 as the "destination" register.
When printing the assembly code with the built-in machine instruction printer,
these destination registers will be printed to the left side of an '=' sign, as
in: %reg1027 = addl %reg1026, %reg1025

This 'addl' MachineInstruction contains three "operands": the first is the
destination register (#1027), the second is the first source register (#1026)
and the third is the second source register (#1025).  Never forget the
destination register will show up in the MachineInstr operands vector.  The code
to generate this instruction looks like this:

  BuildMI(BB, X86::ADDrr32, 2, 1027).addReg(1026).addReg(1025);

The first argument to BuildMI is the basic block to append the machine
instruction to, the second is the opcode, the third is the number of operands,
the fourth is the destination register.  The two addReg calls specify operands
in order.

MachineInstrs that do not produce a value do not have this implicit first
operand, they simply have #operands = #uses.  To create them, simply do not
specify a destination register to the BuildMI call.


======================================
III. Lazy Function Resolution in Jello
======================================

Jello is a designed to be a JIT compiler for LLVM code.  This implies that call
instructions may be emitted before the function they call is compiled.  In order
to support this, Jello currently emits unresolved call instructions to call to a
null pointer.  When the call instruction is executed, a segmentation fault will
be generated.

Jello installs a trap handler for SIGSEGV, in order to trap these events.  When
a SIGSEGV occurs, first we check to see if it's due to lazy function resolution,
if so, we look up the return address of the function call (which was pushed onto
the stack by the call instruction).  Given the return address of the call, we
consult a map to figure out which function was supposed to be called from that
location.

If the function has not been code generated yet, it is at this time.  Finally,
the EIP of the process is modified to point to the real function address, the
original call instruction is updated, and the SIGSEGV handler returns, causing
execution to start in the called function.  Because we update the original call
instruction, we should only get at most one signal for each call site.

Note that this approach does not work for indirect calls.  The problem with
indirect calls is that taking the address of a function would not cause a fault
(it would simply copy null into a register), so we would only find out about the
problem when the indirect call itself was made.  At this point we would have no
way of knowing what the intended function destination was.  Because of this, we
immediately code generate functions whenever they have their address taken,
side-stepping the problem completely.


======================
IV. Source Code Layout
======================

The LLVM-JIT is composed of source files primarily in the following locations:

include/llvm/CodeGen
--------------------
This directory contains header files that are used to represent the program in a
machine specific representation.  It currently also contains a bunch of stuff
used by the Sparc backend that we don't want to get mixed up in, such as
register allocation internals.

include/llvm/Target
-------------------
This directory contains header files that are used to interpret the machine
specific representation of the program.  This allows us to write generic
transformations that will work on any target that implements the interfaces
defined in this directory.  The only classes used by the X86 backend so far are
the TargetMachine, TargetData, MachineInstrInfo, and MRegisterInfo classes.

lib/CodeGen
-----------
This directory will contain all of the target independent transformations (for
example, register allocation) that we write.  These transformations should only
use information exposed through the Target interface, they should not include
any target specific header files.

lib/Target/X86
--------------
This directory contains the machine description for X86 that is required to the
rest of the compiler working.  It contains any code that is truly specific to
the X86 backend, for example the instruction selector and machine code emitter.

tools/lli/JIT
-----------
This directory contains the top-level code for the JIT compiler.  This code
basically boils down to a call to TargetMachine::addPassesToJITCompile.  As we
progress with the project, this will also contain the compile-dispatch-recompile
loop.

test/Regression/Jello
---------------------
This directory contains regression tests for the JIT.  Initially it contains a
bunch of really trivial testcases that we should build up to supporting.


==================================================
V. Strange Things, or, Things That Should Be Known
==================================================

Representing memory in MachineInstrs
------------------------------------

The x86 has a very, uhm, flexible, way of accessing memory.  It is capable of
addressing memory addresses of the following form directly in integer
instructions (which use ModR/M addressing):

   Base+[1,2,4,8]*IndexReg+Disp32

Wow, that's crazy.  In order to represent this, LLVM tracks no less that 4
operands for each memory operand of this form.  This means that the "load" form
of 'mov' has the following "Operands" in this order:

Index:        0     |    1        2       3           4
Meaning:   DestReg, | BaseReg,  Scale, IndexReg, Displacement
OperandTy: VirtReg, | VirtReg, UnsImm, VirtReg,   SignExtImm

Stores and all other instructions treat the four memory operands in the same
way, in the same order.


==========================
VI. TODO / Future Projects
==========================

There are a large number of things remaining to do.  Here is a partial list:

Critical path:
-------------

1. Finish dumb instruction selector

Next Phase:
-----------
1. Implement linear time optimal instruction selector
2. Implement smarter (linear scan?) register allocator

After this project:
-------------------
1. Implement lots of nifty runtime optimizations
2. Implement a static compiler backend for x86 (might come almost for free...)
3. Implement new targets: IA64? X86-64? M68k? MMIX?  Who knows...

Infrastructure Improvements:
----------------------------

1. Bytecode is designed to be able to read particular functions from the
   bytecode without having to read the whole program.  Bytecode reader should be
   extended to allow on-demand loading of functions.

2. PassManager needs to be able to run just a single function through a pipeline
   of FunctionPass's.

3. llvmgcc needs to be modified to output 32-bit little endian LLVM files.
   Preferably it will be parameterizable so that multiple binaries need not
   exist.  Until this happens, we will be restricted to using type safe
   programs (most of the Olden suite and many smaller tests), which should be
   sufficient for our 497 project.  Additionally there are a few places in the
   LLVM infrastructure where we assume Sparc TargetData layout.  These should
   be easy to factor out and identify though.
