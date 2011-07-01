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
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Target/TargetRegistry.h"

#define GET_REGINFO_MC_DESC
#include "X86GenRegisterInfo.inc"

#define GET_INSTRINFO_MC_DESC
#include "X86GenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "X86GenSubtargetInfo.inc"

using namespace llvm;

MCInstrInfo *createX86MCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitX86MCInstrInfo(X);
  return X;
}

MCRegisterInfo *createX86MCRegisterInfo() {
  MCRegisterInfo *X = new MCRegisterInfo();
  InitX86MCRegisterInfo(X);
  return X;
}

MCSubtargetInfo *createX86MCSubtargetInfo() {
  MCSubtargetInfo *X = new MCSubtargetInfo();
  InitX86MCSubtargetInfo(X);
  return X;
}

// Force static initialization.
extern "C" void LLVMInitializeX86MCInstrInfo() {
  RegisterMCInstrInfo<MCInstrInfo> X(TheX86_32Target);
  RegisterMCInstrInfo<MCInstrInfo> Y(TheX86_64Target);

  TargetRegistry::RegisterMCInstrInfo(TheX86_32Target, createX86MCInstrInfo);
  TargetRegistry::RegisterMCInstrInfo(TheX86_64Target, createX86MCInstrInfo);
}

extern "C" void LLVMInitializeX86MCRegInfo() {
  RegisterMCRegInfo<MCRegisterInfo> X(TheX86_32Target);
  RegisterMCRegInfo<MCRegisterInfo> Y(TheX86_64Target);

  TargetRegistry::RegisterMCRegInfo(TheX86_32Target, createX86MCRegisterInfo);
  TargetRegistry::RegisterMCRegInfo(TheX86_64Target, createX86MCRegisterInfo);
}

extern "C" void LLVMInitializeX86MCSubtargetInfo() {
  RegisterMCSubtargetInfo<MCSubtargetInfo> X(TheX86_32Target);
  RegisterMCSubtargetInfo<MCSubtargetInfo> Y(TheX86_64Target);

  TargetRegistry::RegisterMCSubtargetInfo(TheX86_32Target,
                                          createX86MCSubtargetInfo);
  TargetRegistry::RegisterMCSubtargetInfo(TheX86_64Target,
                                          createX86MCSubtargetInfo);
}
