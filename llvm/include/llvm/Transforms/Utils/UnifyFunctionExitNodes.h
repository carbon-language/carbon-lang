//===-- UnifyMethodExitNodes.h - Ensure methods have one return --*- C++ -*--=//
//
// This pass is used to ensure that methods have at most one return instruction
// in them.  It also holds onto the return instruction of the last unified
// method.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_XFORMS_UNIFY_METHOD_EXIT_NODES_H
#define LLVM_XFORMS_UNIFY_METHOD_EXIT_NODES_H

#include "llvm/Pass.h"

struct UnifyMethodExitNodes : public MethodPass {
  BasicBlock *ExitNode;
public:
  static AnalysisID ID;            // Pass ID
  UnifyMethodExitNodes(AnalysisID id = ID) : ExitNode(0) { assert(ID == id); }

  // UnifyAllExitNodes - Unify all exit nodes of the CFG by creating a new
  // BasicBlock, and converting all returns to unconditional branches to this
  // new basic block.  The singular exit node is returned in ExitNode.
  //
  // If there are no return stmts in the Method, a null pointer is returned.
  //
  static bool doit(Function *F, BasicBlock *&ExitNode);


  virtual bool runOnMethod(Function *F) {
    return doit(F, ExitNode);
  }

  BasicBlock *getExitNode() const { return ExitNode; }

  virtual void getAnalysisUsageInfo(Pass::AnalysisSet &Required,
                                    Pass::AnalysisSet &Destroyed,
                                    Pass::AnalysisSet &Provided) {
    // FIXME: Should invalidate CFG
    Provided.push_back(ID);  // Provide self!
  }
};

static inline Pass *createUnifyMethodExitNodesPass() {
  return new UnifyMethodExitNodes();
}

#endif
