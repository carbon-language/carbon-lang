//===- llvm/Transforms/CleanupGCCOutput.h - Cleanup GCC Output ---*- C++ -*--=//
//
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_CLEANUPGCCOUTPUT_H
#define LLVM_TRANSFORMS_CLEANUPGCCOUTPUT_H

#include "llvm/Pass.h"

struct CleanupGCCOutput : public MethodPass {
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

  // getAnalysisUsageInfo - This function needs FindUsedTypes to do its job...
  //
  virtual void getAnalysisUsageInfo(Pass::AnalysisSet &Required,
                                    Pass::AnalysisSet &Destroyed,
                                    Pass::AnalysisSet &Provided);
};

#endif
