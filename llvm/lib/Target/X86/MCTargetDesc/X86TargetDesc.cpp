//===-- X86TargetDesc.cpp - X86 Target Descriptions -------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides X86 specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "X86TargetDesc.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/Target/TargetRegistry.h"

#define GET_REGINFO_MC_DESC
#include "X86GenRegisterInfo.inc"
using namespace llvm;

MCRegisterInfo *createX86MCRegisterInfo() {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitX86MCRegisterInfo(X);
  return X;
}

// Force static initialization.
extern "C" void LLVMInitializeX86MCRegInfo() {
  RegisterMCRegInfo<MCRegisterInfo> X(TheX86_32Target);
  RegisterMCRegInfo<MCRegisterInfo> Y(TheX86_64Target);

  TargetRegistry::RegisterMCRegInfo(TheX86_32Target, createX86MCRegisterInfo);
  TargetRegistry::RegisterMCRegInfo(TheX86_64Target, createX86MCRegisterInfo);
}
