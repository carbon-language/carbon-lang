//===-- DCE.h - Functions that perform Dead Code Elimination -----*- C++ -*--=//
//
// This family of functions is useful for performing dead code elimination of 
// various sorts.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPT_DCE_H
#define LLVM_OPT_DCE_H

#include "llvm/Pass.h"
#include "llvm/Method.h"

//===----------------------------------------------------------------------===//
// DeadInstElimination - This pass quickly removes trivially dead instructions
// without modifying the CFG of the function.  It is a BasicBlockPass, so it
// runs efficiently when queued next to other BasicBlockPass's.
//
struct DeadInstElimination : public BasicBlockPass {
  virtual bool runOnBasicBlock(BasicBlock *BB);
};


//===----------------------------------------------------------------------===//
// DeadCodeElimination - This pass is more powerful than DeadInstElimination,
// because it will remove dead basic blocks as well as all of the instructions
// contained within them.  This pass is useful to run after another pass has
// reorganized the CFG and possibly modified control flow.
//
// TODO: In addition to DCE stuff, this also merges basic blocks together and
// otherwise simplifies control flow.  This should be factored out of this pass
// eventually into it's own pass.
//
struct DeadCodeElimination : public MethodPass {
  // External Interface:
  //
  static bool doDCE(Method *M);

  // dceInstruction - Inspect the instruction at *BBI and figure out if it's
  // [trivially] dead.  If so, remove the instruction and update the iterator
  // to point to the instruction that immediately succeeded the original
  // instruction.
  //
  static bool dceInstruction(BasicBlock::InstListType &BBIL,
                             BasicBlock::iterator &BBI);

  // Remove unused global values - This removes unused global values of no
  // possible value.  This currently includes unused method prototypes and
  // unitialized global variables.
  //
  static bool RemoveUnusedGlobalValues(Module *M);

  // Pass Interface...
  virtual bool doInitialization(Module *M) {
    return RemoveUnusedGlobalValues(M);
  }
  virtual bool runOnMethod(Method *M) { return doDCE(M); }
  virtual bool doFinalization(Module *M) {
    return RemoveUnusedGlobalValues(M);
  }
};



//===----------------------------------------------------------------------===//
// AgressiveDCE - This pass uses the SSA based Agressive DCE algorithm.  This
// algorithm assumes instructions are dead until proven otherwise, which makes
// it more successful are removing non-obviously dead instructions.
//
struct AgressiveDCE : public MethodPass {
  virtual bool runOnMethod(Method *M);

  // getAnalysisUsageInfo - We require post dominance frontiers (aka Control
  // Dependence Graph)
  virtual void getAnalysisUsageInfo(Pass::AnalysisSet &Requires,
                                    Pass::AnalysisSet &Destroyed,
                                    Pass::AnalysisSet &Provided);
};


// SimplifyCFG - This function is used to do simplification of a CFG.  For
// example, it adjusts branches to branches to eliminate the extra hop, it
// eliminates unreachable basic blocks, and does other "peephole" optimization
// of the CFG.  It returns true if a modification was made, and returns an 
// iterator that designates the first element remaining after the block that
// was deleted.
//
// WARNING:  The entry node of a method may not be simplified.
//
bool SimplifyCFG(Method::iterator &BBIt);

#endif
