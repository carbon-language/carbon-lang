//===- llvm/Transforms/CleanupGCCOutput.h - Cleanup GCC Output ---*- C++ -*--=//
//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_CLEANUPGCCOUTPUT_H
#define LLVM_TRANSFORMS_CLEANUPGCCOUTPUT_H

#include "llvm/Pass.h"

class CleanupGCCOutput : public Pass {
public:
  // doPassInitialization - For this pass, it removes global symbol table
  // entries for primitive types.  These are never used for linking in GCC and
  // they make the output uglier to look at, so we nuke them.
  //
  bool doPassInitialization(Module *M);

  // doPerMethodWork - This method simplifies the specified method hopefully.
  //
  bool doPerMethodWork(Method *M);
};

#endif
