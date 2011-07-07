//===- SystemZSubtarget.cpp - SystemZ Subtarget Information -------*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SystemZ specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "SystemZSubtarget.h"
#include "SystemZ.h"
#include "llvm/GlobalValue.h"
#include "llvm/Target/TargetMachine.h"

#define GET_SUBTARGETINFO_CTOR
#define GET_SUBTARGETINFO_MC_DESC
#define GET_SUBTARGETINFO_TARGET_DESC
#include "SystemZGenSubtargetInfo.inc"

using namespace llvm;

SystemZSubtarget::SystemZSubtarget(const std::string &TT, 
                                   const std::string &CPU,
                                   const std::string &FS):
  SystemZGenSubtargetInfo(TT, CPU, FS), HasZ10Insts(false) {
  std::string CPUName = CPU;
  if (CPUName.empty())
    CPUName = "z9";

  // Parse features string.
  ParseSubtargetFeatures(CPUName, FS);
}

/// True if accessing the GV requires an extra load.
bool SystemZSubtarget::GVRequiresExtraLoad(const GlobalValue* GV,
                                           const TargetMachine& TM,
                                           bool isDirectCall) const {
  if (TM.getRelocationModel() == Reloc::PIC_) {
    // Extra load is needed for all externally visible.
    if (isDirectCall)
      return false;

    if (GV->hasLocalLinkage() || GV->hasHiddenVisibility())
      return false;

    return true;
  }

  return false;
}
