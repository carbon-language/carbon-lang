//===- Linker.cpp - Module Linker Implementation --------------------------===//
//
// This file implements the LLVM module linker.
//
// Specifically, this:
//  - Merges global variables between the two modules
//    - Uninit + Uninit = Init, Init + Uninit = Init, Init + Init = Error if !=
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Linker.h"


// LinkModules - This function links two modules together, with the resulting
// left module modified to be the composite of the two input modules.  If an
// error occurs, true is returned and ErrorMsg (if not null) is set to indicate
// the problem.
//
bool LinkModules(Module *Dest, const Module *Src, string *ErrorMsg = 0) {

  return false;
}
