//===-- DCE.h - Functions that perform Dead Code Elimination -----*- C++ -*--=//
//
// This family of functions is useful for performing dead code elimination of 
// various sorts.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_OPT_DCE_H
#define LLVM_OPT_DCE_H

#include "llvm/Module.h"
#include "llvm/Method.h"

namespace opt {

bool DoDeadCodeElimination(Method *M);         // DCE a method
bool DoRemoveUnusedConstants(SymTabValue *S);  // RUC a method or module
bool DoDeadCodeElimination(Module *C);         // DCE & RUC a whole module


// DoADCE - Execute the Agressive Dead Code Elimination Algorithm
//
bool DoADCE(Method *M);                        // Defined in ADCE.cpp
static inline bool DoADCE(Module *M) {
  return M->reduceApply(DoADCE);
}

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
