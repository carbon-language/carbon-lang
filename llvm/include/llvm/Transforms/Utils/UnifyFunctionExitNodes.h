//===-- UnifyFunctionExitNodes.h - Ensure fn's have one return ---*- C++ -*--=//
//
// This pass is used to ensure that functions have at most one return
// instruction in them.  Additionally, it keeps track of which node is the new
// exit node of the CFG.  If there are no exit nodes in the CFG, the getExitNode
// method will return a null pointer.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_UNIFYFUNCTIONEXITNODES_H
#define LLVM_TRANSFORMS_UNIFYFUNCTIONEXITNODES_H

#include "llvm/Pass.h"

struct UnifyFunctionExitNodes : public FunctionPass {
  BasicBlock *ExitNode;
public:
  UnifyFunctionExitNodes() : ExitNode(0) {}

  // We can preserve non-critical-edgeness when we unify function exit nodes
  virtual void getAnalysisUsage(AnalysisUsage &AU) const;

  // getExitNode - Return the new single (or nonexistant) exit node of the CFG.
  //
  BasicBlock *getExitNode() const { return ExitNode; }

  virtual bool runOnFunction(Function &F);
};

static inline Pass *createUnifyFunctionExitNodesPass() {
  return new UnifyFunctionExitNodes();
}

#endif
