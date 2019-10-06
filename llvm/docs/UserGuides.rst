User Guides
===========

NOTE: If you are a user who is only interested in using an LLVM-based compiler,
you should look into `Clang <http://clang.llvm.org>`_ instead. The
documentation here is intended for users who have a need to work with the
intermediate LLVM representation.

.. contents::
   :local:

.. toctree::
   :hidden:

   CMake
   CMakePrimer
   AdvancedBuilds
   HowToBuildOnARM
   HowToBuildWithPGO
   HowToCrossCompileBuiltinsOnArm
   HowToCrossCompileLLVM
   yaml2obj
   MarkdownQuickstartTemplate
   Phabricator
   Passes
   MCJITDesignAndImplementation
   ORCv2
   CodeOfConduct
   CompileCudaWithLLVM
   ReportingGuide
   Benchmarking
   Docker
   BuildingADistribution
   Remarks
   WritingAnLLVMPass
   WritingAnLLVMBackend
   TableGen/index
   NVPTXUsage
   AMDGPUUsage
   ExtendingLLVM
   CommandLine

Clang
-----

:doc:`HowToBuildOnARM`
   Notes on building and testing LLVM/Clang on ARM.

:doc:`HowToBuildWithPGO`
    Notes on building LLVM/Clang with PGO.

:doc:`HowToCrossCompileLLVM`
   Notes on cross-building and testing LLVM/Clang.

`How to build the C, C++, ObjC, and ObjC++ front end`__
   Instructions for building the clang front-end from source.

   .. __: http://clang.llvm.org/get_started.html

LLVM Builds and Distributions
-----------------------------

:doc:`BuildingADistribution`
  A best-practices guide for using LLVM's CMake build system to package and
  distribute LLVM-based tools.

:doc:`CMake`
   An addendum to the main Getting Started guide for those using the `CMake
   build system <http://www.cmake.org>`_.

:doc:`Docker`
   A reference for using Dockerfiles provided with LLVM.

Optimizations
-------------

:doc:`WritingAnLLVMPass`
   Information on how to write LLVM transformations and analyses.

:doc:`Passes`
   A list of optimizations and analyses implemented in LLVM.

:doc:`LoopTerminology`
  A document describing Loops and associated terms as used in LLVM.

:doc:`Remarks`
   A reference on the implementation of remarks in LLVM.

Code Generation
---------------

:doc:`WritingAnLLVMBackend`
   Information on how to write LLVM backends for machine targets.

:doc:`CodeGenerator`
   The design and implementation of the LLVM code generator.  Useful if you are
   working on retargetting LLVM to a new architecture, designing a new codegen
   pass, or enhancing existing components.

:doc:`TableGen <TableGen/index>`
   Describes the TableGen tool, which is used heavily by the LLVM code
   generator.

Additional Topics
-----------------

:doc:`CommandLine`
  Provides information on using the command line parsing library.

:doc:`ExtendingLLVM`
  Look here to see how to add instructions and intrinsics to LLVM.

:doc:`HowToCrossCompileBuiltinsOnArm`
   Notes on cross-building and testing the compiler-rt builtins for Arm.

:doc:`NVPTXUsage`
   This document describes using the NVPTX backend to compile GPU kernels.

:doc:`AMDGPUUsage`
   This document describes using the AMDGPU backend to compile GPU kernels.