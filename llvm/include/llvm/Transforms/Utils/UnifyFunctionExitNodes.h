//===-- UnifyFunctionExitNodes.h - Ensure fn's have one return --*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This pass is used to ensure that functions have at most one return and one
// unwind instruction in them.  Additionally, it keeps track of which node is
// the new exit node of the CFG.  If there are no return or unwind instructions
// in the function, the getReturnBlock/getUnwindBlock methods will return a null
// pointer.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UNIFYFUNCTIONEXITNODES_H
#define LLVM_TRANSFORMS_UNIFYFUNCTIONEXITNODES_H

#include "llvm/Pass.h"

namespace llvm {

struct UnifyFunctionExitNodes : public FunctionPass {
  BasicBlock *ReturnBlock, *UnwindBlock;
public:
  UnifyFunctionExitNodes() : ReturnBlock(0), UnwindBlock(0) {}

  // We can preserve non-critical-edgeness when we unify function exit nodes
  virtual void getAnalysisUsage(AnalysisUsage &AU) const;

  // getReturn|UnwindBlock - Return the new single (or nonexistant) return or
  // unwind basic blocks in the CFG.
  //
  BasicBlock *getReturnBlock() const { return ReturnBlock; }
  BasicBlock *getUnwindBlock() const { return UnwindBlock; }

  virtual bool runOnFunction(Function &F);
};

Pass *createUnifyFunctionExitNodesPass();

} // End llvm namespace

#endif
