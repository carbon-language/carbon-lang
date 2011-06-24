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
#include "X86GenRegisterDesc.inc"
using namespace llvm;

MCRegisterInfo *createX86MCRegisterInfo() {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitX86MCRegisterInfo(X);
  return X;
}
