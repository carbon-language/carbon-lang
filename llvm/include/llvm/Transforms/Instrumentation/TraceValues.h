// $Id$ -*-c++-*-
//***************************************************************************
// File:
//	TraceValues.h
// 
// Purpose:
//      Support for inserting LLVM code to print values at basic block
//      and method exits.  Also exports functions to create a call
//      "printf" instruction with one of the signatures listed below.
// 
// History:
//	10/11/01	 -  Vikram Adve  -  Created
//**************************************************************************/

#ifndef LLVM_TRANSFORMS_INSTRUMENTATION_TRACEVALUES_H
#define LLVM_TRANSFORMS_INSTRUMENTATION_TRACEVALUES_H

#include "llvm/Transforms/Pass.h"

class InsertTraceCode : public Pass {
  bool TraceBasicBlockExits, TraceMethodExits;
public:
  InsertTraceCode(bool traceBasicBlockExits, bool traceMethodExits)
    : TraceBasicBlockExits(traceBasicBlockExits), 
      TraceMethodExits(traceMethodExits) {}


  //--------------------------------------------------------------------------
  // Function InsertCodeToTraceValues
  // 
  // Inserts tracing code for all live values at basic block and/or method exits
  // as specified by `traceBasicBlockExits' and `traceMethodExits'.
  //--------------------------------------------------------------------------

  static bool doInsertTraceCode(Method *M, bool traceBasicBlockExits,
                                bool traceMethodExits);



  // doPerMethodWork - This method does the work.  Always successful.
  //
  bool doPerMethodWork(Method *M) {
    return doInsertTraceCode(M, TraceBasicBlockExits, TraceMethodExits);
  }
};

#endif /*LLVM_TRANSFORMS_INSTRUMENTATION_TRACEVALUES_H*/
