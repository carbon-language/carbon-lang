//===-- UnifyFunctionExitNodes.h - Ensure fn's have one return ---*- C++ -*--=//
//
// This pass is used to ensure that functions have at most one return
// instruction in them.  It also holds onto the return instruction of the last
// unified function.
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

  virtual const char *getPassName() const { return "Unify Function Exit Nodes";}

  // UnifyAllExitNodes - Unify all exit nodes of the CFG by creating a new
  // BasicBlock, and converting all returns to unconditional branches to this
  // new basic block.  The singular exit node is returned in ExitNode.
  //
  // If there are no return stmts in the function, a null pointer is returned.
  //
  static bool doit(Function *F, BasicBlock *&ExitNode);


  virtual bool runOnFunction(Function *F) {
    return doit(F, ExitNode);
  }

  BasicBlock *getExitNode() const { return ExitNode; }

  virtual void getAnalysisUsage(AnalysisUsage &AU) const {
    AU.addProvided(ID);  // Provide self!
  }
};

static inline Pass *createUnifyFunctionExitNodesPass() {
  return new UnifyFunctionExitNodes();
}

#endif
