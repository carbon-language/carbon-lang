//===- llvm/Transforms/Instrumentation/TraceValues.h - Tracing ---*- C++ -*--=//
//
// Support for inserting LLVM code to print values at basic block and method
// exits.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_TRACEVALUES_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_TRACEVALUES_H

#include "llvm/Pass.h"
class Method;

class InsertTraceCode : public MethodPass {
  bool TraceBasicBlockExits, TraceMethodExits;
  Method *PrintfMeth;
public:
  InsertTraceCode(bool traceBasicBlockExits, bool traceMethodExits)
    : TraceBasicBlockExits(traceBasicBlockExits), 
      TraceMethodExits(traceMethodExits) {}

  // Add a prototype for printf if it is not already in the program.
  //
  bool doInitialization(Module *M);

  //--------------------------------------------------------------------------
  // Function InsertCodeToTraceValues
  // 
  // Inserts tracing code for all live values at basic block and/or method exits
  // as specified by `traceBasicBlockExits' and `traceMethodExits'.
  //
  static bool doit(Method *M, bool traceBasicBlockExits,
                   bool traceMethodExits, Method *Printf);

  // runOnMethod - This method does the work.  Always successful.
  //
  bool runOnMethod(Method *M) {
    return doit(M, TraceBasicBlockExits, TraceMethodExits, PrintfMeth);
  }
};

#endif
