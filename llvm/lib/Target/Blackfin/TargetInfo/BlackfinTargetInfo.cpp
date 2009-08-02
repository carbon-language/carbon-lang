//===-- BlackfinTargetInfo.cpp - Blackfin Target Implementation -----------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "Blackfin.h"
#include "llvm/Module.h"
#include "llvm/Target/TargetRegistry.h"

using namespace llvm;

Target llvm::TheBlackfinTarget;

extern "C" void LLVMInitializeBlackfinTargetInfo() {
  RegisterTarget<Triple::bfin> X(TheBlackfinTarget, "bfin",
                                 "Analog Devices Blackfin [experimental]");
}
