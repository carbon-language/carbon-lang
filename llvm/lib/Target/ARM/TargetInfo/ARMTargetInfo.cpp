//===-- ARMTargetInfo.cpp - ARM Target Implementation ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "MCTargetDesc/ARMMCTargetDesc.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/TargetRegistry.h"
using namespace llvm;

Target llvm::TheARMleTarget,   llvm::TheARMbeTarget;
Target llvm::TheThumbleTarget, llvm::TheThumbbeTarget;

extern "C" void LLVMInitializeARMTargetInfo() { 
  RegisterTarget<Triple::arm, /*HasJIT=*/true>
    X(TheARMleTarget, "arm", "ARM");
  RegisterTarget<Triple::armeb, /*HasJIT=*/true>
    Y(TheARMbeTarget, "armeb", "ARM (big endian)");

  RegisterTarget<Triple::thumb, /*HasJIT=*/true>
    A(TheThumbleTarget, "thumb", "Thumb");
  RegisterTarget<Triple::thumbeb, /*HasJIT=*/true>
    B(TheThumbbeTarget, "thumbeb", "Thumb (big endian)");
}
