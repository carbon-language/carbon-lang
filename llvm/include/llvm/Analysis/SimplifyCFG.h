//===- llvm/Analysis/SimplifyCFG.h - CFG Simplification XForms ---*- C++ -*--=//
//
// This file provides several routines that are useful for simplifying CFGs in
// various ways...
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_SIMPLIFY_CFG_H
#define LLVM_ANALYSIS_SIMPLIFY_CFG_H

class BasicBlock;
class Method;

namespace cfg {

  // UnifyAllExitNodes - Unify all exit nodes of the CFG by creating a new
  // BasicBlock, and converting all returns to unconditional branches to this
  // new basic block.  The singular exit node is returned.
  //
  // If there are no return stmts in the Method, a null pointer is returned.
  //
  BasicBlock *UnifyAllExitNodes(Method *M);

}  // End Namespace cfg


#endif
