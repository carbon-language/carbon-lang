.. _subsystems:

Subsystem Documentation
=======================

.. toctree::
   :hidden:

   AliasAnalysis
   LinkTimeOptimization

* `Writing an LLVM Pass <WritingAnLLVMPass.html>`_
    
   Information on how to write LLVM transformations and analyses.
    
* `Writing an LLVM Backend <WritingAnLLVMBackend.html>`_
    
   Information on how to write LLVM backends for machine targets.
    
* `The LLVM Target-Independent Code Generator <CodeGenerator.html>`_
    
   The design and implementation of the LLVM code generator.  Useful if you are
   working on retargetting LLVM to a new architecture, designing a new codegen
   pass, or enhancing existing components.
    
* `TableGen Fundamentals <TableGenFundamentals.html>`_
    
   Describes the TableGen tool, which is used heavily by the LLVM code
   generator.
    
* :ref:`alias_analysis`
    
   Information on how to write a new alias analysis implementation or how to
   use existing analyses.
    
* `Accurate Garbage Collection with LLVM <GarbageCollection.html>`_
    
   The interfaces source-language compilers should use for compiling GC'd
   programs.
    
* `Source Level Debugging with LLVM <SourceLevelDebugging.html>`_
    
   This document describes the design and philosophy behind the LLVM
   source-level debugger.
    
* `Zero Cost Exception handling in LLVM <ExceptionHandling.html>`_
    
   This document describes the design and implementation of exception handling
   in LLVM.
    
* `Bugpoint <Bugpoint.html>`_
    
   Automatic bug finder and test-case reducer description and usage
   information.
    
* `LLVM Bitcode File Format <BitCodeFormat.html>`_
    
   This describes the file format and encoding used for LLVM "bc" files.
    
* `System Library <SystemLibrary.html>`_
    
   This document describes the LLVM System Library (<tt>lib/System</tt>) and
   how to keep LLVM source code portable
    
* :ref:`lto`
    
   This document describes the interface between LLVM intermodular optimizer
   and the linker and its design
    
* `The LLVM gold plugin <GoldPlugin.html>`_
    
   How to build your programs with link-time optimization on Linux.
    
* `The GDB JIT interface <DebuggingJITedCode.html>`_
    
   How to debug JITed code with GDB.
    
* `Branch Weight Metadata <BranchWeightMetadata.html>`_
    
   Provides information about Branch Prediction Information.
