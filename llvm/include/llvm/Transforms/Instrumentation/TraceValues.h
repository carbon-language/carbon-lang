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

class Instruction;
class Value;
class Type;


//************************** External Functions ****************************/


//--------------------------------------------------------------------------
// Function GetPrintMethodForType
// 
// Creates an external declaration for "printf".
// The signatures supported are:
//   int printf(sbyte*,  sbyte*,  sbyte*,  sbyte*,  int      intValue)
//   int printf(sbyte*,  sbyte*,  sbyte*,  sbyte*,  unsigned uintValue)
//   int printf(sbyte*,  sbyte*,  sbyte*,  sbyte*,  float    floatValue)
//   int printf(sbyte*,  sbyte*,  sbyte*,  sbyte*,  double   doubleValue)
//   int printf(sbyte*,  sbyte*,  sbyte*,  sbyte*,  char*    stringValue)
//   int printf(sbyte*,  sbyte*,  sbyte*,  sbyte*,  void*    ptrValue)
// 
// The invocation should be:
//       call "printf"(fmt, bbName, valueName, valueType, value).
//--------------------------------------------------------------------------

const Method*	GetPrintMethodForType   (Module* module,
                                         Type* vtype);


//--------------------------------------------------------------------------
// Function CreatePrintInstr
// 
// Creates an invocation of printf for the value `val' at the exit of the
// basic block `bb'.  isMethodExit specifies if this is a method exit, 
//--------------------------------------------------------------------------

Instruction*	CreatePrintInstr        (Value* val,
                                         const BasicBlock* bb,
                                         Module* module,
                                         unsigned int indent,
                                         bool isMethodExit);

//--------------------------------------------------------------------------
// Function InsertCodeToTraceValues
// 
// Inserts tracing code for all live values at basic block and/or method exits
// as specified by `traceBasicBlockExits' and `traceMethodExits'.
//--------------------------------------------------------------------------

void            InsertCodeToTraceValues (Method* method,
                                         bool traceBasicBlockExits,
                                         bool traceMethodExits);


class InsertTraceCode : public ConcretePass {
  bool TraceBasicBlockExits, TraceMethodExits;
public:
  InsertTraceCode(bool traceBasicBlockExits, bool traceMethodExits)
    : TraceBasicBlockExits(traceBasicBlockExits), 
      TraceMethodExits(traceMethodExits) {}

  // doPerMethodWork - This method does the work.  Always successful.
  //
  bool doPerMethodWorkVirt(Method *M) {
    InsertCodeToTraceValues(M, TraceBasicBlockExits, TraceMethodExits);
    return false;
  }
};

#endif /*LLVM_TRANSFORMS_INSTRUMENTATION_TRACEVALUES_H*/
