//===-- FunctionInlining.h - Functions that perform Inlining -----*- C++ -*--=//
//
// This family of functions is useful for performing function inlining.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_FUNCTION_INLINING_H
#define LLVM_TRANSFORMS_FUNCTION_INLINING_H

class CallInst;
class Pass;

Pass *createFunctionInliningPass();

// InlineFunction - This function forcibly inlines the called function into the
// basic block of the caller.  This returns true if it is not possible to inline
// this call.  The program is still in a well defined state if this occurs 
// though.
//
// Note that this only does one level of inlining.  For example, if the 
// instruction 'call B' is inlined, and 'B' calls 'C', then the call to 'C' now 
// exists in the instruction stream.  Similiarly this will inline a recursive
// function by one level.
//
bool InlineFunction(CallInst *C);

#endif
