//===- TestPasses.cpp - "buggy" passes used to test bugpoint --------------===//
//
// This file contains "buggy" passes that are used to test bugpoint, to check
// that it is narrowing down testcases correctly.
//
//===----------------------------------------------------------------------===//

#include "llvm/Pass.h"
#include "llvm/iOther.h"
#include "llvm/Support/InstVisitor.h"
#include "llvm/Constant.h"
#include "llvm/BasicBlock.h"

namespace {
  /// CrashOnCalls - This pass is used to test bugpoint.  It intentionally
  /// crashes on any call instructions.
  class CrashOnCalls : public BasicBlockPass {
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesAll();
    }

    bool runOnBasicBlock(BasicBlock &BB) {
      for (BasicBlock::iterator I = BB.begin(), E = BB.end(); I != E; ++I)
        if (isa<CallInst>(*I))
          abort();

      return false;
    }
  };

  RegisterPass<CrashOnCalls>
  X("bugpoint-crashcalls",
    "BugPoint Test Pass - Intentionally crash on CallInsts");
}

namespace {
  /// DeleteCalls - This pass is used to test bugpoint.  It intentionally
  /// deletes all call instructions, "misoptimizing" the program.
  class DeleteCalls : public BasicBlockPass {
    bool runOnBasicBlock(BasicBlock &BB) {
      for (BasicBlock::iterator I = BB.begin(), E = BB.end(); I != E; ++I)
        if (CallInst *CI = dyn_cast<CallInst>(I)) {
          if (!CI->use_empty())
            CI->replaceAllUsesWith(Constant::getNullValue(CI->getType()));
          CI->getParent()->getInstList().erase(CI);
        }
      return false;
    }
  };

  RegisterPass<DeleteCalls>
  Y("bugpoint-deletecalls",
    "BugPoint Test Pass - Intentionally 'misoptimize' CallInsts");
}
