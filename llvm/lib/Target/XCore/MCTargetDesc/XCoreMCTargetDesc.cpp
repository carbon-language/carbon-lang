//===-- XCoreMCTargetDesc.cpp - XCore Target Descriptions -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file provides XCore specific target descriptions.
//
//===----------------------------------------------------------------------===//

#include "XCoreMCTargetDesc.h"
#include "XCoreMCAsmInfo.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/Target/TargetRegistry.h"

#define GET_INSTRINFO_MC_DESC
#include "XCoreGenInstrInfo.inc"

#define GET_SUBTARGETINFO_MC_DESC
#include "XCoreGenSubtargetInfo.inc"

#define GET_REGINFO_MC_DESC
#include "XCoreGenRegisterInfo.inc"

using namespace llvm;

static MCInstrInfo *createXCoreMCInstrInfo() {
  MCInstrInfo *X = new MCInstrInfo();
  InitXCoreMCInstrInfo(X);
  return X;
}

extern "C" void LLVMInitializeXCoreMCInstrInfo() {
  TargetRegistry::RegisterMCInstrInfo(TheXCoreTarget, createXCoreMCInstrInfo);
}

static MCSubtargetInfo *createXCoreMCSubtargetInfo(StringRef TT, StringRef CPU,
                                                   StringRef FS) {
  MCSubtargetInfo *X = new MCSubtargetInfo();
  InitXCoreMCSubtargetInfo(X, TT, CPU, FS);
  return X;
}

extern "C" void LLVMInitializeXCoreMCSubtargetInfo() {
  TargetRegistry::RegisterMCSubtargetInfo(TheXCoreTarget,
                                          createXCoreMCSubtargetInfo);
}

extern "C" void LLVMInitializeXCoreMCAsmInfo() {
  RegisterMCAsmInfo<XCoreMCAsmInfo> X(TheXCoreTarget);
}
