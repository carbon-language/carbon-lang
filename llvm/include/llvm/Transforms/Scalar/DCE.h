//===-- DCE.h - Functions that perform Dead Code Elimination -----*- C++ -*--=//
//
// This family of functions is useful for performing dead code elimination of 
// various sorts.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPT_DCE_H
#define LLVM_OPT_DCE_H

#include "llvm/Transforms/Pass.h"

namespace opt {

struct DeadCodeElimination : public Pass {
  // External Interface:
  //
  static bool doDCE(Method *M);

  // Remove unused global values - This removes unused global values of no
  // possible value.  This currently includes unused method prototypes and
  // unitialized global variables.
  //
  static bool RemoveUnusedGlobalValues(Module *M);

  // RemoveUnusedGlobalValuesAfterLink - This function is only to be used after
  // linking the application.  It removes global variables with initializers and
  // unreachable methods.  This should only be used after an application is
  // linked, when it is not possible for an external entity to make a global
  // value live again.
  //
  // static bool RemoveUnusedGlobalValuesAfterLink(Module *M); // TODO

  // Pass Interface...
  virtual bool doPassInitialization(Module *M) {
    return RemoveUnusedGlobalValues(M);
  }
  virtual bool doPerMethodWork(Method *M) { return doDCE(M); }
  virtual bool doPassFinalization(Module *M) {
    return RemoveUnusedGlobalValues(M);
  }
};



struct AgressiveDCE : public Pass {
  // DoADCE - Execute the Agressive Dead Code Elimination Algorithm
  //
  static bool doADCE(Method *M);                        // Defined in ADCE.cpp

  virtual bool doPerMethodWork(Method *M) {
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

}  // End namespace opt

#endif
