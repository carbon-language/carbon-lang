//===-- PIC16TargetInfo.cpp - PIC16 Target Implementation -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "PIC16.h"
#include "llvm/Module.h"
#include "llvm/Target/TargetRegistry.h"
using namespace llvm;

Target llvm::ThePIC16Target, llvm::TheCooperTarget;

extern "C" void LLVMInitializePIC16TargetInfo() { 
  RegisterTarget<> X(ThePIC16Target, "pic16", "PIC16 14-bit [experimental]");

  RegisterTarget<> Y(TheCooperTarget, "cooper", "PIC16 Cooper [experimental]");
}
