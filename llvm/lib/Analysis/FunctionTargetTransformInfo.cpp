//===- llvm/Analysis/FunctionTargetTransformInfo.h --------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass wraps a TargetTransformInfo in a FunctionPass so that it can
// forward along the current Function so that we can make target specific
// decisions based on the particular subtarget specified for each Function.
//
//===----------------------------------------------------------------------===//

#include "llvm/InitializePasses.h"
#include "llvm/Analysis/FunctionTargetTransformInfo.h"

using namespace llvm;

#define DEBUG_TYPE "function-tti"
static const char ftti_name[] = "Function TargetTransformInfo";
INITIALIZE_PASS_BEGIN(FunctionTargetTransformInfo, "function_tti", ftti_name, false, true)
INITIALIZE_AG_DEPENDENCY(TargetTransformInfo)
INITIALIZE_PASS_END(FunctionTargetTransformInfo, "function_tti", ftti_name, false, true)
char FunctionTargetTransformInfo::ID = 0;

namespace llvm {
FunctionPass *createFunctionTargetTransformInfoPass() {
  return new FunctionTargetTransformInfo();
}
}

FunctionTargetTransformInfo::FunctionTargetTransformInfo()
  : FunctionPass(ID), Fn(nullptr), TTI(nullptr) {
  initializeFunctionTargetTransformInfoPass(*PassRegistry::getPassRegistry());
}

void FunctionTargetTransformInfo::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  AU.addRequired<TargetTransformInfo>();
}

void FunctionTargetTransformInfo::releaseMemory() {}

bool FunctionTargetTransformInfo::runOnFunction(Function &F) {
  Fn = &F;
  TTI = &getAnalysis<TargetTransformInfo>();
  return false;
}
