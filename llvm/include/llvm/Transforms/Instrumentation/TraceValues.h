//===- llvm/Transforms/Instrumentation/TraceValues.h - Tracing ---*- C++ -*--=//
//
// Support for inserting LLVM code to print values at basic block and method
// exits.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_TRACEVALUES_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_TRACEVALUES_H

class Pass;
Pass *createTraceValuesPassForMethod();       // Just trace methods
Pass *createTraceValuesPassForBasicBlocks();  // Trace BB's and methods

#endif
