//===- llvm/Transforms/CleanupGCCOutput.h - Cleanup GCC Output ---*- C++ -*--=//
//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_CLEANUPGCCOUTPUT_H
#define LLVM_TRANSFORMS_CLEANUPGCCOUTPUT_H

#include "llvm/Analysis/FindUsedTypes.h"

class CleanupGCCOutput : public MethodPass {
  Method *Malloc, *Free;  // Pointers to external declarations, or null if none
  FindUsedTypes FUT;      // Use FUT to eliminate type names that are never used
public:

  inline CleanupGCCOutput() : Malloc(0), Free(0) {}

  // PatchUpMethodReferences - This is a part of the functionality exported by
  // the CleanupGCCOutput pass.  This causes functions with different signatures
  // to be linked together if they have the same name.
  //
  static bool PatchUpMethodReferences(Module *M);

  // doPassInitialization - For this pass, it removes global symbol table
  // entries for primitive types.  These are never used for linking in GCC and
  // they make the output uglier to look at, so we nuke them.
  //
  // Also, initialize instance variables.
  //
  bool doInitialization(Module *M);

  // doPerMethodWork - This method simplifies the specified method hopefully.
  //
  bool runOnMethod(Method *M);

  // doPassFinalization - Strip out type names that are unused by the program
  bool doFinalization(Module *M);
private:
  bool doOneCleanupPass(Method *M);
};

#endif
