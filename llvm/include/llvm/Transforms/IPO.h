//===- llvm/Transforms/CleanupGCCOutput.h - Cleanup GCC Output ---*- C++ -*--=//
//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_CLEANUPGCCOUTPUT_H
#define LLVM_TRANSFORMS_CLEANUPGCCOUTPUT_H

#include "llvm/Pass.h"

class CleanupGCCOutput : public Pass {
  Method *Malloc, *Free;  // Pointers to external declarations, or null if none
public:

  inline CleanupGCCOutput() : Malloc(0), Free(0) {}

  // doPassInitialization - For this pass, it removes global symbol table
  // entries for primitive types.  These are never used for linking in GCC and
  // they make the output uglier to look at, so we nuke them.
  //
  // Also, initialize instance variables.
  //
  bool doPassInitialization(Module *M);

  // doPerMethodWork - This method simplifies the specified method hopefully.
  //
  bool doPerMethodWork(Method *M);
private:
  bool doOneCleanupPass(Method *M);
};

#endif
