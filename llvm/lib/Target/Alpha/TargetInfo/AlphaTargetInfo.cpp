//===-- AlphaTargetInfo.cpp - Alpha Target Implementation -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Alpha.h"
#include "llvm/Module.h"
#include "llvm/Target/TargetRegistry.h"
using namespace llvm;

llvm::Target llvm::TheAlphaTarget;

extern "C" void LLVMInitializeAlphaTargetInfo() { 
  RegisterTarget<Triple::alpha, /*HasJIT=*/true>
    X(TheAlphaTarget, "alpha", "Alpha [experimental]");
}
