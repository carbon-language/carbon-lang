//===-- XCoreTargetInfo.cpp - XCore Target Implementation -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "XCore.h"
#include "llvm/Module.h"
#include "llvm/Target/TargetRegistry.h"
using namespace llvm;

Target llvm::TheXCoreTarget;

static unsigned XCore_TripleMatchQuality(const std::string &TT) {
  if (TT.size() >= 6 && std::string(TT.begin(), TT.begin()+6) == "xcore-")
    return 20;

  return 0;
}

static unsigned XCore_ModuleMatchQuality(const Module &M) {
  // Check for a triple match.
  if (unsigned Q = XCore_TripleMatchQuality(M.getTargetTriple()))
    return Q;

  // Otherwise we don't match.
  return 0;
}

extern "C" void LLVMInitializeXCoreTargetInfo() { 
  TargetRegistry::RegisterTarget(TheXCoreTarget, "xcore",
                                  "XCore",
                                  &XCore_TripleMatchQuality,
                                  &XCore_ModuleMatchQuality);
}
