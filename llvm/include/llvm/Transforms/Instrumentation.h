//===- Transforms/Instrumentation.h - Instrumentation passes ----*- C++ -*-===//
//
// This files defines constructor functions for instrumentation passes.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_H

class Pass;

//===----------------------------------------------------------------------===//
// Support for inserting LLVM code to print values at basic block and function
// exits.
//
Pass *createTraceValuesPassForFunction();     // Just trace function entry/exit
Pass *createTraceValuesPassForBasicBlocks();  // Trace BB's and methods

#endif
