Subsystem Documentation
=======================

.. toctree::
   :hidden:

   AliasAnalysis
   BitCodeFormat
   BranchWeightMetadata
   Bugpoint
   CodeGenerator
   ExceptionHandling
   LinkTimeOptimization
   SegmentedStacks
   TableGenFundamentals
   DebuggingJITedCode
   GoldPlugin
   MarkedUpDisassembly
   SystemLibrary
   SourceLevelDebugging
   Vectorizers
   WritingAnLLVMBackend
   GarbageCollection
   WritingAnLLVMPass
   TableGen/LangRef

* :doc:`WritingAnLLVMPass`

   Information on how to write LLVM transformations and analyses.

* :doc:`WritingAnLLVMBackend`

   Information on how to write LLVM backends for machine targets.

* :doc:`CodeGenerator`

   The design and implementation of the LLVM code generator.  Useful if you are
   working on retargetting LLVM to a new architecture, designing a new codegen
   pass, or enhancing existing components.
    
* :doc:`TableGenFundamentals`

   Describes the TableGen tool, which is used heavily by the LLVM code
   generator.
    
* :doc:`AliasAnalysis`
    
   Information on how to write a new alias analysis implementation or how to
   use existing analyses.

* :doc:`GarbageCollection`

   The interfaces source-language compilers should use for compiling GC'd
   programs.

* :doc:`Source Level Debugging with LLVM <SourceLevelDebugging>`
    
   This document describes the design and philosophy behind the LLVM
   source-level debugger.

* :doc:`Vectorizers`
    
   This document describes the current status of vectorization in LLVM.
    
* :doc:`ExceptionHandling`
    
   This document describes the design and implementation of exception handling
   in LLVM.
    
* :doc:`Bugpoint`
    
   Automatic bug finder and test-case reducer description and usage
   information.
    
* :doc:`BitCodeFormat`
    
   This describes the file format and encoding used for LLVM "bc" files.
    
* :doc:`System Library <SystemLibrary>`
    
   This document describes the LLVM System Library (``lib/System``) and
   how to keep LLVM source code portable
    
* :doc:`LinkTimeOptimization`
    
   This document describes the interface between LLVM intermodular optimizer
   and the linker and its design
    
* :doc:`GoldPlugin`
    
   How to build your programs with link-time optimization on Linux.
    
* :doc:`DebuggingJITedCode`
    
   How to debug JITed code with GDB.
    
* :doc:`BranchWeightMetadata`
    
   Provides information about Branch Prediction Information.

* :doc:`SegmentedStacks`

   This document describes segmented stacks and how they are used in LLVM.

* :doc:`MarkedUpDisassembly`

   This document describes the optional rich disassembly output syntax.

