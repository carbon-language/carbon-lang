//===- README.txt - Information about the X86 backend and related files ---===//
//
// This file contains random notes and points of interest about the X86 backend.
//
//===----------------------------------------------------------------------===//

===========
I. Overview
===========

This directory contains a machine description for the X86 processor family.
Currently this machine description is used for a high performance code generator
used by the LLVM JIT and static code generators.  One of the main objectives
that we would like to support with this project is to build a nice clean code
generator that may be extended in the future in a variety of ways: new targets,
new optimizations, new transformations, etc.

This document describes the current state of the X86 code generator, along with
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
agnostic, representing instructions in their most abstract form: an opcode and a
series of operands.  This representation is designed to support both SSA
representation for machine code, as well as a register allocated, non-SSA form.

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
X86InstrInfo.td file.  In the X86 backend, there are two particularly
interesting forms of machine instruction: those that produce a value (such as
add), and those that do not (such as a store).

Instructions that produce a value use Operand #0 as the "destination" register.
When printing the assembly code with the built-in machine instruction printer,
these destination registers will be printed to the left side of an '=' sign, as
in: %reg1027 = add %reg1026, %reg1025

This `add' MachineInstruction contains three "operands": the first is the
destination register (#1027), the second is the first source register (#1026)
and the third is the second source register (#1025).  Never forget the
destination register will show up in the MachineInstr operands vector.  The code
to generate this instruction looks like this:

  BuildMI(BB, X86::ADD32rr, 2, 1027).addReg(1026).addReg(1025);

The first argument to BuildMI is the basic block to append the machine
instruction to, the second is the opcode, the third is the number of operands,
the fourth is the destination register.  The two addReg calls specify operands
in order.

MachineInstrs that do not produce a value do not have this implicit first
operand, they simply have #operands = #uses.  To create them, simply do not
specify a destination register to the BuildMI call.


======================
IV. Source Code Layout
======================

The LLVM code generator is composed of source files primarily in the following
locations:

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

lib/ExecutionEngine/JIT
-----------------------
This directory contains the top-level code for the JIT compiler.  This code
basically boils down to a call to TargetMachine::addPassesToJITCompile, and
handles the compile-dispatch-recompile cycle.

test/Regression/CodeGen/X86
---------------------------
This directory contains regression tests for the X86 code generator.


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


======================
VI. Instruction naming
======================

An instruction name consists of the base name, a default operand size
followed by a character per operand with an optional special size. For
example:

ADD8rr -> add, 8-bit register, 8-bit register

IMUL16rmi -> imul, 16-bit register, 16-bit memory, 16-bit immediate

IMUL16rmi8 -> imul, 16-bit register, 16-bit memory, 8-bit immediate

MOVSX32rm16 -> movsx, 32-bit register, 16-bit memory


==========================
VII. TODO / Future Projects
==========================

Ideas for Improvements:
-----------------------
1. Implement an *optimal* linear time instruction selector
2. Implement lots of nifty runtime optimizations
3. Implement new targets: IA64? X86-64? M68k? MMIX?  Who knows...

Infrastructure Improvements:
----------------------------

1. X86/Printer.cpp and Sparc/EmitAssembly.cpp both have copies of what is
   roughly the same code, used to output constants in a form the assembler
   can understand. These functions should be shared at some point. They
   should be rewritten to pass around iostreams instead of strings. The
   list of functions is as follows:

   isStringCompatible
   toOctal
   ConstantExprToString
   valToExprString
   getAsCString
   printSingleConstantValue (with TypeToDataDirective inlined)
   printConstantValueOnly
