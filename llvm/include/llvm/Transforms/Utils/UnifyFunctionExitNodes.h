//===-- UnifyFunctionExitNodes.h - Ensure fn's have one return ---*- C++ -*--=//
//
// This pass is used to ensure that functions have at most one return
// instruction in them.  Additionally, it keeps track of which node is the new
// exit node of the CFG.  If there are no exit nodes in the CFG, the getExitNode
// method will return a null pointer.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_XFORMS_UNIFY_FUNCTION_EXIT_NODES_H
#define LLVM_XFORMS_UNIFY_FUNCTION_EXIT_NODES_H

#include "llvm/Pass.h"

struct UnifyFunctionExitNodes : public FunctionPass {
  BasicBlock *ExitNode;
public:
  static AnalysisID ID;            // Pass ID
  UnifyFunctionExitNodes(AnalysisID id = ID) : ExitNode(0) { assert(ID == id); }

  // getExitNode - Return the new single (or nonexistant) exit node of the CFG.
  //
  BasicBlock *getExitNode() const { return ExitNode; }

  virtual const char *getPassName() const { return "Unify Function Exit Nodes";}
  virtual bool runOnFunction(Function &F);
  virtual void getAnalysisUsage(AnalysisUsage &AU) const { AU.addProvided(ID); }
};

static inline Pass *createUnifyFunctionExitNodesPass() {
  return new UnifyFunctionExitNodes();
}

#endif
