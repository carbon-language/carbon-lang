//===-- DCE.h - Functions that perform Dead Code Elimination -----*- C++ -*--=//
//
// This family of functions is useful for performing dead code elimination of 
// various sorts.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPT_DCE_H
#define LLVM_OPT_DCE_H

#include "llvm/Pass.h"
#include "llvm/BasicBlock.h"

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



struct AgressiveDCE : public MethodPass {
  // DoADCE - Execute the Agressive Dead Code Elimination Algorithm
  //
  static bool doADCE(Method *M);                        // Defined in ADCE.cpp

  virtual bool runOnMethod(Method *M) {
    return doADCE(M);
  }
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
