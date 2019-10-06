Reference
=========

LLVM and API reference documentation.

.. contents::
   :local:

.. toctree::
   :hidden:

   Atomics
   Bugpoint
   CommandGuide/index
   CompilerWriterInfo
   Extensions
   FuzzingLLVM
   GarbageCollection
   GetElementPtr
   GwpAsan
   HowToSetUpLLVMStyleRTTI
   LangRef
   LibFuzzer
   MIRLangRef
   OptBisect
   PDB/index
   ScudoHardenedAllocator
   Statepoints
   TestingGuide
   YamlIO

API Reference
-------------

`Doxygen generated documentation <http://llvm.org/doxygen/>`_
  (`classes <http://llvm.org/doxygen/inherits.html>`_)

`Documentation for Go bindings <http://godoc.org/llvm.org/llvm/bindings/go/llvm>`_

LLVM Reference
--------------

:doc:`LLVM Language Reference Manual <LangRef>`
  Defines the LLVM intermediate representation and the assembly form of the
  different nodes.

:doc:`Machine IR (MIR) Format Reference Manual <MIRLangRef>`
   A reference manual for the MIR serialization format, which is used to test
   LLVM's code generation passes.

:doc:`Atomics`
  Information about LLVM's concurrency model.

:doc:`CompilerWriterInfo`
  A list of helpful links for compiler writers.

:doc:`Extensions`
  LLVM-specific extensions to tools and formats LLVM seeks compatibility with.

:doc:`HowToSetUpLLVMStyleRTTI`
  How to make ``isa<>``, ``dyn_cast<>``, etc. available for clients of your
  class hierarchy.

:doc:`GetElementPtr`
  Answers to some very frequent questions about LLVM's most frequently
  misunderstood instruction.

:doc:`ScudoHardenedAllocator`
  A library that implements a security-hardened `malloc()`.

:doc:`GwpAsan`
  A sampled heap memory error detection toolkit designed for production use.

:doc:`YamlIO`
   A reference guide for using LLVM's YAML I/O library.

======================
Command Line Utilities
======================

:doc:`LLVM Command Guide <CommandGuide/index>`
   A reference manual for the LLVM command line utilities ("man" pages for LLVM
   tools).

:doc:`Bugpoint`
   Automatic bug finder and test-case reducer description and usage
   information.

:doc:`OptBisect`
  A command line option for debugging optimization-induced failures.

:doc:`The Microsoft PDB File Format <PDB/index>`
  A detailed description of the Microsoft PDB (Program Database) file format.

==================
Garbage Collection
==================

:doc:`GarbageCollection`
   The interfaces source-language compilers should use for compiling GC'd
   programs.

:doc:`Statepoints`
  This describes a set of experimental extensions for garbage
  collection support.

=========
LibFuzzer
=========

:doc:`LibFuzzer`
  A library for writing in-process guided fuzzers.

:doc:`FuzzingLLVM`
  Information on writing and using Fuzzers to find bugs in LLVM.

=======
Testing
=======

:doc:`LLVM Testing Infrastructure Guide <TestingGuide>`
   A reference manual for using the LLVM testing infrastructure.

:doc:`TestSuiteGuide`
  Describes how to compile and run the test-suite benchmarks.