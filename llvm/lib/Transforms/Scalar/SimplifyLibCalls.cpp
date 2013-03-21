//===- SimplifyLibCalls.cpp - Optimize specific well-known library calls --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements a simple pass that applies a variety of small
// optimizations for calls to specific well-known function calls (e.g. runtime
// library functions).   Any optimization that takes the very simple form
// "replace call to library function with simpler code that provides the same
// result" belongs in this file.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "simplify-libcalls"
#include "llvm/Transforms/Scalar.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Analysis/ValueTracking.h"
#include "llvm/Config/config.h"            // FIXME: Shouldn't depend on host!
#include "llvm/IR/DataLayout.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Target/TargetLibraryInfo.h"
#include "llvm/Transforms/Utils/BuildLibCalls.h"
using namespace llvm;


//===----------------------------------------------------------------------===//
// Optimizer Base Class
//===----------------------------------------------------------------------===//

/// This class is the abstract base class for the set of optimizations that
/// corresponds to one library call.
namespace {
class LibCallOptimization {
protected:
  Function *Caller;
  const DataLayout *TD;
  const TargetLibraryInfo *TLI;
  LLVMContext* Context;
public:
  LibCallOptimization() { }
  virtual ~LibCallOptimization() {}

  /// CallOptimizer - This pure virtual method is implemented by base classes to
  /// do various optimizations.  If this returns null then no transformation was
  /// performed.  If it returns CI, then it transformed the call and CI is to be
  /// deleted.  If it returns something else, replace CI with the new value and
  /// delete CI.
  virtual Value *CallOptimizer(Function *Callee, CallInst *CI, IRBuilder<> &B)
    =0;

  Value *OptimizeCall(CallInst *CI, const DataLayout *TD,
                      const TargetLibraryInfo *TLI, IRBuilder<> &B) {
    Caller = CI->getParent()->getParent();
    this->TD = TD;
    this->TLI = TLI;
    if (CI->getCalledFunction())
      Context = &CI->getCalledFunction()->getContext();

    // We never change the calling convention.
    if (CI->getCallingConv() != llvm::CallingConv::C)
      return NULL;

    return CallOptimizer(CI->getCalledFunction(), CI, B);
  }
};
} // End anonymous namespace.


//===----------------------------------------------------------------------===//
// SimplifyLibCalls Pass Implementation
//===----------------------------------------------------------------------===//

namespace {
  /// This pass optimizes well known library functions from libc and libm.
  ///
  class SimplifyLibCalls : public FunctionPass {
    TargetLibraryInfo *TLI;

    StringMap<LibCallOptimization*> Optimizations;

    bool Modified;  // This is only used by doInitialization.
  public:
    static char ID; // Pass identification
    SimplifyLibCalls() : FunctionPass(ID) {
      initializeSimplifyLibCallsPass(*PassRegistry::getPassRegistry());
    }
    void AddOpt(LibFunc::Func F, LibCallOptimization* Opt);
    void AddOpt(LibFunc::Func F1, LibFunc::Func F2, LibCallOptimization* Opt);

    void InitOptimizations();
    bool runOnFunction(Function &F);

    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.addRequired<TargetLibraryInfo>();
    }
  };
} // end anonymous namespace.

char SimplifyLibCalls::ID = 0;

INITIALIZE_PASS_BEGIN(SimplifyLibCalls, "simplify-libcalls",
                      "Simplify well-known library calls", false, false)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfo)
INITIALIZE_PASS_END(SimplifyLibCalls, "simplify-libcalls",
                    "Simplify well-known library calls", false, false)

// Public interface to the Simplify LibCalls pass.
FunctionPass *llvm::createSimplifyLibCallsPass() {
  return new SimplifyLibCalls();
}

void SimplifyLibCalls::AddOpt(LibFunc::Func F, LibCallOptimization* Opt) {
  if (TLI->has(F))
    Optimizations[TLI->getName(F)] = Opt;
}

void SimplifyLibCalls::AddOpt(LibFunc::Func F1, LibFunc::Func F2,
                              LibCallOptimization* Opt) {
  if (TLI->has(F1) && TLI->has(F2))
    Optimizations[TLI->getName(F1)] = Opt;
}

/// Optimizations - Populate the Optimizations map with all the optimizations
/// we know.
void SimplifyLibCalls::InitOptimizations() {
}


/// runOnFunction - Top level algorithm.
///
bool SimplifyLibCalls::runOnFunction(Function &F) {
  TLI = &getAnalysis<TargetLibraryInfo>();

  if (Optimizations.empty())
    InitOptimizations();

  const DataLayout *TD = getAnalysisIfAvailable<DataLayout>();

  IRBuilder<> Builder(F.getContext());

  bool Changed = false;
  for (Function::iterator BB = F.begin(), E = F.end(); BB != E; ++BB) {
    for (BasicBlock::iterator I = BB->begin(), E = BB->end(); I != E; ) {
      // Ignore non-calls.
      CallInst *CI = dyn_cast<CallInst>(I++);
      if (!CI || CI->hasFnAttr(Attribute::NoBuiltin)) continue;

      // Ignore indirect calls and calls to non-external functions.
      Function *Callee = CI->getCalledFunction();
      if (Callee == 0 || !Callee->isDeclaration() ||
          !(Callee->hasExternalLinkage() || Callee->hasDLLImportLinkage()))
        continue;

      // Ignore unknown calls.
      LibCallOptimization *LCO = Optimizations.lookup(Callee->getName());
      if (!LCO) continue;

      // Set the builder to the instruction after the call.
      Builder.SetInsertPoint(BB, I);

      // Use debug location of CI for all new instructions.
      Builder.SetCurrentDebugLocation(CI->getDebugLoc());

      // Try to optimize this call.
      Value *Result = LCO->OptimizeCall(CI, TD, TLI, Builder);
      if (Result == 0) continue;

      DEBUG(dbgs() << "SimplifyLibCalls simplified: " << *CI;
            dbgs() << "  into: " << *Result << "\n");

      // Something changed!
      Changed = true;

      // Inspect the instruction after the call (which was potentially just
      // added) next.
      I = CI; ++I;

      if (CI != Result && !CI->use_empty()) {
        CI->replaceAllUsesWith(Result);
        if (!Result->hasName())
          Result->takeName(CI);
      }
      CI->eraseFromParent();
    }
  }
  return Changed;
}

// TODO:
//   Additional cases that we need to add to this file:
//
// cbrt:
//   * cbrt(expN(X))  -> expN(x/3)
//   * cbrt(sqrt(x))  -> pow(x,1/6)
//   * cbrt(sqrt(x))  -> pow(x,1/9)
//
// exp, expf, expl:
//   * exp(log(x))  -> x
//
// log, logf, logl:
//   * log(exp(x))   -> x
//   * log(x**y)     -> y*log(x)
//   * log(exp(y))   -> y*log(e)
//   * log(exp2(y))  -> y*log(2)
//   * log(exp10(y)) -> y*log(10)
//   * log(sqrt(x))  -> 0.5*log(x)
//   * log(pow(x,y)) -> y*log(x)
//
// lround, lroundf, lroundl:
//   * lround(cnst) -> cnst'
//
// pow, powf, powl:
//   * pow(exp(x),y)  -> exp(x*y)
//   * pow(sqrt(x),y) -> pow(x,y*0.5)
//   * pow(pow(x,y),z)-> pow(x,y*z)
//
// round, roundf, roundl:
//   * round(cnst) -> cnst'
//
// signbit:
//   * signbit(cnst) -> cnst'
//   * signbit(nncst) -> 0 (if pstv is a non-negative constant)
//
// sqrt, sqrtf, sqrtl:
//   * sqrt(expN(x))  -> expN(x*0.5)
//   * sqrt(Nroot(x)) -> pow(x,1/(2*N))
//   * sqrt(pow(x,y)) -> pow(|x|,y*0.5)
//
// strchr:
//   * strchr(p, 0) -> strlen(p)
// tan, tanf, tanl:
//   * tan(atan(x)) -> x
//
// trunc, truncf, truncl:
//   * trunc(cnst) -> cnst'
//
//
