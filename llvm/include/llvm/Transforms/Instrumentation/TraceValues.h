//===- llvm/Transforms/Instrumentation/TraceValues.h - Tracing ---*- C++ -*--=//
//
// Support for inserting LLVM code to print values at basic block and function
// exits.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_TRACEVALUES_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_TRACEVALUES_H

class Pass;
Pass *createTraceValuesPassForFunction();     // Just trace function entry/exit
Pass *createTraceValuesPassForBasicBlocks();  // Trace BB's and methods

#endif
