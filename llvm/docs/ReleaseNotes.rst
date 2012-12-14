.. raw:: html

    <style> .red {color:red} </style>

.. role:: red

======================
LLVM 3.2 Release Notes
======================

.. contents::
    :local:

Written by the `LLVM Team <http://llvm.org/>`_

:red:`These are in-progress notes for the upcoming LLVM 3.2 release.  You may
prefer the` `LLVM 3.1 Release Notes <http://llvm.org/releases/3.1/docs
/ReleaseNotes.html>`_.

Introduction
============

This document contains the release notes for the LLVM Compiler Infrastructure,
release 3.2.  Here we describe the status of LLVM, including major improvements
from the previous release, improvements in various subprojects of LLVM, and
some of the current users of the code.  All LLVM releases may be downloaded
from the `LLVM releases web site <http://llvm.org/releases/>`_.

For more information about LLVM, including information about the latest
release, please check out the `main LLVM web site <http://llvm.org/>`_.  If you
have questions or comments, the `LLVM Developer's Mailing List
<http://lists.cs.uiuc.edu/mailman/listinfo/llvmdev>`_ is a good place to send
them.

Note that if you are reading this file from a Subversion checkout or the main
LLVM web page, this document applies to the *next* release, not the current
one.  To see the release notes for a specific release, please see the `releases
page <http://llvm.org/releases/>`_.

Sub-project Status Update
=========================

The LLVM 3.2 distribution currently consists of code from the core LLVM
repository, which roughly includes the LLVM optimizers, code generators and
supporting tools, and the Clang repository.  In addition to this code, the LLVM
Project includes other sub-projects that are in development.  Here we include
updates on these subprojects.

Clang: C/C++/Objective-C Frontend Toolkit
-----------------------------------------

`Clang <http://clang.llvm.org/>`_ is an LLVM front end for the C, C++, and
Objective-C languages.  Clang aims to provide a better user experience through
expressive diagnostics, a high level of conformance to language standards, fast
compilation, and low memory use.  Like LLVM, Clang provides a modular,
library-based architecture that makes it suitable for creating or integrating
with other development tools.  Clang is considered a production-quality
compiler for C, Objective-C, C++ and Objective-C++ on x86 (32- and 64-bit), and
for Darwin/ARM targets.

In the LLVM 3.2 time-frame, the Clang team has made many improvements.
Highlights include:

#. More powerful warnings, especially `-Wuninitialized`
#. Template type diffing in diagnostic messages
#. Higher quality and more efficient debug info generation

For more details about the changes to Clang since the 3.1 release, see the
`Clang release notes. <http://clang.llvm.org/docs/ReleaseNotes.html>`_

If Clang rejects your code but another compiler accepts it, please take a look
at the `language compatibility <http://clang.llvm.org/compatibility.html>`_
guide to make sure this is not intentional or a known issue.

DragonEgg: GCC front-ends, LLVM back-end
----------------------------------------

`DragonEgg <http://dragonegg.llvm.org/>`_ is a `gcc plugin
<http://gcc.gnu.org/wiki/plugins>`_ that replaces GCC's optimizers and code
generators with LLVM's.  It works with gcc-4.5 and gcc-4.6 (and partially with
gcc-4.7), can target the x86-32/x86-64 and ARM processor families, and has been
successfully used on the Darwin, FreeBSD, KFreeBSD, Linux and OpenBSD
platforms.  It fully supports Ada, C, C++ and Fortran.  It has partial support
for Go, Java, Obj-C and Obj-C++.

The 3.2 release has the following notable changes:

#. ...

compiler-rt: Compiler Runtime Library
-------------------------------------

The new LLVM `compiler-rt project <http://compiler-rt.llvm.org/>`_ is a simple
library that provides an implementation of the low-level target-specific hooks
required by code generation and other runtime components.  For example, when
compiling for a 32-bit target, converting a double to a 64-bit unsigned integer
is compiled into a runtime call to the ``__fixunsdfdi`` function.  The
``compiler-rt`` library provides highly optimized implementations of this and
other low-level routines (some are 3x faster than the equivalent libgcc
routines).

The 3.2 release has the following notable changes:

#. ...

LLDB: Low Level Debugger
------------------------

`LLDB <http://lldb.llvm.org>`_ is a ground-up implementation of a command line
debugger, as well as a debugger API that can be used from other applications.
LLDB makes use of the Clang parser to provide high-fidelity expression parsing
(particularly for C++) and uses the LLVM JIT for target support.

The 3.2 release has the following notable changes:

#. ...

libc++: C++ Standard Library
----------------------------

Like compiler_rt, libc++ is now :ref:`dual licensed
<copyright-license-patents>` under the MIT and UIUC license, allowing it to be
used more permissively.

Within the LLVM 3.2 time-frame there were the following highlights:

#. ...

VMKit
-----

The `VMKit project <http://vmkit.llvm.org/>`_ is an implementation of a Java
Virtual Machine (Java VM or JVM) that uses LLVM for static and just-in-time
compilation.

The 3.2 release has the following notable changes:

#. ...

Polly: Polyhedral Optimizer
---------------------------

`Polly <http://polly.llvm.org/>`_ is an *experimental* optimizer for data
locality and parallelism.  It provides high-level loop optimizations and
automatic parallelisation.

Within the LLVM 3.2 time-frame there were the following highlights:

#. isl, the integer set library used by Polly, was relicensed to the MIT license
#. isl based code generation
#. MIT licensed replacement for CLooG (LGPLv2)
#. Fine grained option handling (separation of core and border computations,
   control overhead vs. code size)
#. Support for FORTRAN and dragonegg
#. OpenMP code generation fixes

External Open Source Projects Using LLVM 3.2
============================================

An exciting aspect of LLVM is that it is used as an enabling technology for a
lot of other language and tools projects.  This section lists some of the
projects that have already been updated to work with LLVM 3.2.

Crack
-----

`Crack <http://code.google.com/p/crack-language/>`_ aims to provide the ease of
development of a scripting language with the performance of a compiled
language.  The language derives concepts from C++, Java and Python,
incorporating object-oriented programming, operator overloading and strong
typing.

FAUST
-----

`FAUST <http://faust.grame.fr/>`_ is a compiled language for real-time audio
signal processing.  The name FAUST stands for Functional AUdio STream.  Its
programming model combines two approaches: functional programming and block
diagram composition.  In addition with the C, C++, Java, JavaScript output
formats, the Faust compiler can generate LLVM bitcode, and works with LLVM
2.7-3.1.

Glasgow Haskell Compiler (GHC)
------------------------------

`GHC <http://www.haskell.org/ghc/>`_ is an open source compiler and programming
suite for Haskell, a lazy functional programming language.  It includes an
optimizing static compiler generating good code for a variety of platforms,
together with an interactive system for convenient, quick development.

GHC 7.0 and onwards include an LLVM code generator, supporting LLVM 2.8 and
later.

Julia
-----

`Julia <https://github.com/JuliaLang/julia>`_ is a high-level, high-performance
dynamic language for technical computing.  It provides a sophisticated
compiler, distributed parallel execution, numerical accuracy, and an extensive
mathematical function library.  The compiler uses type inference to generate
fast code without any type declarations, and uses LLVM's optimization passes
and JIT compiler.  The `Julia Language <http://julialang.org/>`_ is designed
around multiple dispatch, giving programs a large degree of flexibility.  It is
ready for use on many kinds of problems.

LLVM D Compiler
---------------

`LLVM D Compiler <https://github.com/ldc-developers/ldc>`_ (LDC) is a compiler
for the D programming Language.  It is based on the DMD frontend and uses LLVM
as backend.

Open Shading Language
---------------------

`Open Shading Language (OSL)
<https://github.com/imageworks/OpenShadingLanguage/>`_ is a small but rich
language for programmable shading in advanced global illumination renderers and
other applications, ideal for describing materials, lights, displacement, and
pattern generation.  It uses LLVM to JIT complex shader networks to x86 code at
runtime.

OSL was developed by Sony Pictures Imageworks for use in its in-house renderer
used for feature film animation and visual effects, and is distributed as open
source software with the "New BSD" license.

Portable OpenCL (pocl)
----------------------

In addition to producing an easily portable open source OpenCL implementation,
another major goal of `pocl <http://pocl.sourceforge.net/>`_ is improving
performance portability of OpenCL programs with compiler optimizations,
reducing the need for target-dependent manual optimizations.  An important part
of pocl is a set of LLVM passes used to statically parallelize multiple
work-items with the kernel compiler, even in the presence of work-group
barriers.  This enables static parallelization of the fine-grained static
concurrency in the work groups in multiple ways (SIMD, VLIW, superscalar, ...).

Pure
----

`Pure <http://pure-lang.googlecode.com/>`_ is an algebraic/functional
programming language based on term rewriting.  Programs are collections of
equations which are used to evaluate expressions in a symbolic fashion.  The
interpreter uses LLVM as a backend to JIT-compile Pure programs to fast native
code.  Pure offers dynamic typing, eager and lazy evaluation, lexical closures,
a hygienic macro system (also based on term rewriting), built-in list and
matrix support (including list and matrix comprehensions) and an easy-to-use
interface to C and other programming languages (including the ability to load
LLVM bitcode modules, and inline C, C++, Fortran and Faust code in Pure
programs if the corresponding LLVM-enabled compilers are installed).

Pure version 0.54 has been tested and is known to work with LLVM 3.1 (and
continues to work with older LLVM releases >= 2.5).

TTA-based Co-design Environment (TCE)
-------------------------------------

`TCE <http://tce.cs.tut.fi/>`_ is a toolset for designing application-specific
processors (ASP) based on the Transport triggered architecture (TTA).  The
toolset provides a complete co-design flow from C/C++ programs down to
synthesizable VHDL/Verilog and parallel program binaries.  Processor
customization points include the register files, function units, supported
operations, and the interconnection network.

TCE uses Clang and LLVM for C/C++ language support, target independent
optimizations and also for parts of code generation.  It generates new
LLVM-based code generators "on the fly" for the designed TTA processors and
loads them in to the compiler backend as runtime libraries to avoid per-target
recompilation of larger parts of the compiler chain.

Installation Instructions
=========================

See :doc:`GettingStarted`.

What's New in LLVM 3.2?
=======================

This release includes a huge number of bug fixes, performance tweaks and minor
improvements.  Some of the major improvements and new features are listed in
this section.

Major New Features
------------------

..

  Features that need text if they're finished for 3.2:
   ARM EHABI
   combiner-aa?
   strong phi elim
   loop dependence analysis
   CorrelatedValuePropagation
   Integrated assembler on by default for arm/thumb?

  Near dead:
   Analysis/RegionInfo.h + Dom Frontiers
   SparseBitVector: used in LiveVar.
   llvm/lib/Archive - replace with lib object?


LLVM 3.2 includes several major changes and big features:

#. New NVPTX back-end (replacing existing PTX back-end) based on NVIDIA sources
#. ...

LLVM IR and Core Improvements
-----------------------------

LLVM IR has several new features for better support of new targets and that
expose new optimization opportunities:

#. Thread local variables may have a specified TLS model.  See the :ref:`Language
   Reference Manual <globalvars>`.
#. ...

Optimizer Improvements
----------------------

In addition to many minor performance tweaks and bug fixes, this release
includes a few major enhancements and additions to the optimizers:

Loop Vectorizer - We've added a loop vectorizer and we are now able to
vectorize small loops.  The loop vectorizer is disabled by default and can be
enabled using the ``-mllvm -vectorize-loops`` flag.  The SIMD vector width can
be specified using the flag ``-mllvm -force-vector-width=4``.  The default
value is ``0`` which means auto-select.

We can now vectorize this function:

.. code-block:: c++

  unsigned sum_arrays(int *A, int *B, int start, int end) {
    unsigned sum = 0;
    for (int i = start; i < end; ++i)
      sum += A[i] + B[i] + i;
    return sum;
  }

We vectorize under the following loops:

#. The inner most loops must have a single basic block.
#. The number of iterations are known before the loop starts to execute.
#. The loop counter needs to be incremented by one.
#. The loop trip count **can** be a variable.
#. Loops do **not** need to start at zero.
#. The induction variable can be used inside the loop.
#. Loop reductions are supported.
#. Arrays with affine access pattern do **not** need to be marked as
   '``noalias``' and are checked at runtime.
#. ...

SROA - We've re-written SROA to be significantly more powerful and generate
code which is much more friendly to the rest of the optimization pipeline.
Previously this pass had scaling problems that required it to only operate on
relatively small aggregates, and at times it would mistakenly replace a large
aggregate with a single very large integer in order to make it a scalar SSA
value. The result was a large number of i1024 and i2048 values representing any
small stack buffer. These in turn slowed down many subsequent optimization
paths.

The new SROA pass uses a different algorithm that allows it to only promote to
scalars the pieces of the aggregate actively in use. Because of this it doesn't
require any thresholds. It also always deduces the scalar values from the uses
of the aggregate rather than the specific LLVM type of the aggregate. These
features combine to both optimize more code with the pass but to improve the
compile time of many functions dramatically.

#. Branch weight metadata is preseved through more of the optimizer.
#. ...

MC Level Improvements
---------------------

The LLVM Machine Code (aka MC) subsystem was created to solve a number of
problems in the realm of assembly, disassembly, object file format handling,
and a number of other related areas that CPU instruction-set level tools work
in.  For more information, please see the `Intro to the LLVM MC Project Blog
Post <http://blog.llvm.org/2010/04/intro-to-llvm-mc-project.html>`_.

#. ...

.. _codegen:

Target Independent Code Generator Improvements
----------------------------------------------

We have put a significant amount of work into the code generator
infrastructure, which allows us to implement more aggressive algorithms and
make it run faster:

#. ...

Stack Coloring - We have implemented a new optimization pass to merge stack
objects which are used in disjoin areas of the code.  This optimization reduces
the required stack space significantly, in cases where it is clear to the
optimizer that the stack slot is not shared.  We use the lifetime markers to
tell the codegen that a certain alloca is used within a region.

We now merge consecutive loads and stores.

X86-32 and X86-64 Target Improvements
-------------------------------------

New features and major changes in the X86 target include:

#. ...

.. _ARM:

ARM Target Improvements
-----------------------

New features of the ARM target include:

#. ...

.. _armintegratedassembler:

MIPS Target Improvements
------------------------

New features and major changes in the MIPS target include:

#. ...

PowerPC Target Improvements
---------------------------

Many fixes and changes across LLVM (and Clang) for better compliance with the
64-bit PowerPC ELF Application Binary Interface, interoperability with GCC, and
overall 64-bit PowerPC support.  Some highlights include:

#. MCJIT support added.
#. PPC64 relocation support and (small code model) TOC handling added.
#. Parameter passing and return value fixes (alignment issues, padding, varargs
   support, proper register usage, odd-sized structure support, float support,
   extension of return values for i32 return values).
#. Fixes in spill and reload code for vector registers.
#. C++ exception handling enabled.
#. Changes to remediate double-rounding compatibility issues with respect to
   GCC behavior.
#. Refactoring to disentangle ``ppc64-elf-linux`` ABI from Darwin ppc64 ABI
   support.
#. Assorted new test cases and test case fixes (endian and word size issues).
#. Fixes for big-endian codegen bugs, instruction encodings, and instruction
   constraints.
#. Implemented ``-integrated-as`` support.
#. Additional support for Altivec compare operations.
#. IBM long double support.

There have also been code generation improvements for both 32- and 64-bit code.
Instruction scheduling support for the Freescale e500mc and e5500 cores has
been added.

PTX/NVPTX Target Improvements
-----------------------------

The PTX back-end has been replaced by the NVPTX back-end, which is based on the
LLVM back-end used by NVIDIA in their CUDA (nvcc) and OpenCL compiler.  Some
highlights include:

#. Compatibility with PTX 3.1 and SM 3.5.
#. Support for NVVM intrinsics as defined in the NVIDIA Compiler SDK.
#. Full compatibility with old PTX back-end, with much greater coverage of LLVM
   SIR.

Please submit any back-end bugs to the LLVM Bugzilla site.

Other Target Specific Improvements
----------------------------------

#. ...

Major Changes and Removed Features
----------------------------------

If you're already an LLVM user or developer with out-of-tree changes based on
LLVM 3.2, this section lists some "gotchas" that you may run into upgrading
from the previous release.

#. The CellSPU port has been removed.  It can still be found in older versions.
#. ...

Internal API Changes
--------------------

In addition, many APIs have changed in this release.  Some of the major LLVM
API changes are:

We've added a new interface for allowing IR-level passes to access
target-specific information.  A new IR-level pass, called
``TargetTransformInfo`` provides a number of low-level interfaces.  LSR and
LowerInvoke already use the new interface.

The ``TargetData`` structure has been renamed to ``DataLayout`` and moved to
``VMCore`` to remove a dependency on ``Target``.

#. ...

Tools Changes
-------------

In addition, some tools have changed in this release.  Some of the changes are:

#. ...

Python Bindings
---------------

Officially supported Python bindings have been added!  Feature support is far
from complete.  The current bindings support interfaces to:

#. ...

Known Problems
==============

LLVM is generally a production quality compiler, and is used by a broad range
of applications and shipping in many products.  That said, not every subsystem
is as mature as the aggregate, particularly the more obscure1 targets.  If you
run into a problem, please check the `LLVM bug database
<http://llvm.org/bugs/>`_ and submit a bug if there isn't already one or ask on
the `LLVMdev list <http://lists.cs.uiuc.edu/mailman/listinfo/llvmdev>`_.

Known problem areas include:

#. MSP430, and XCore backends are experimental.

#. The integrated assembler, disassembler, and JIT is not supported by several
   targets.  If an integrated assembler is not supported, then a system
   assembler is required.  For more details, see the
   :ref:`target-feature-matrix`.

Additional Information
======================

A wide variety of additional information is available on the `LLVM web page
<http://llvm.org/>`_, in particular in the `documentation
<http://llvm.org/docs/>`_ section.  The web page also contains versions of the
API documentation which is up-to-date with the Subversion version of the source
code.  You can access versions of these documents specific to this release by
going into the ``llvm/docs/`` directory in the LLVM tree.

If you have any questions or comments about LLVM, please feel free to contact
us via the `mailing lists <http://llvm.org/docs/#maillist>`_.

