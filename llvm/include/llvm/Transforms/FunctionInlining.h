//===-- MethodInlining.h - Functions that perform Inlining -------*- C++ -*--=//
//
// This family of functions is useful for performing method inlining.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPT_METHOD_INLINING_H
#define LLVM_OPT_METHOD_INLINING_H

#include "llvm/Transforms/Pass.h"
#include "llvm/BasicBlock.h"
class CallInst;

namespace opt {

struct MethodInlining : public StatelessPass<MethodInlining> {
  // DoMethodInlining - Use a heuristic based approach to inline methods that
  // seem to look good.
  //
  static bool doMethodInlining(Method *M);

  inline static bool doPerMethodWork(Method *M) {
    return doMethodInlining(M);
  }
};

// InlineMethod - This function forcibly inlines the called method into the
// basic block of the caller.  This returns true if it is not possible to inline
// this call.  The program is still in a well defined state if this occurs 
// though.
//
// Note that this only does one level of inlining.  For example, if the 
// instruction 'call B' is inlined, and 'B' calls 'C', then the call to 'C' now 
// exists in the instruction stream.  Similiarly this will inline a recursive
// method by one level.
//
bool InlineMethod(CallInst *C);
bool InlineMethod(BasicBlock::iterator CI);  // *CI must be CallInst

}  // end namespace opt

#endif
