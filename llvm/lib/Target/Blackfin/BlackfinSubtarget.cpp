//===- BlackfinSubtarget.cpp - BLACKFIN Subtarget Information -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the blackfin specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "BlackfinSubtarget.h"
#include "Blackfin.h"
#include "llvm/Target/TargetRegistry.h"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "BlackfinGenSubtargetInfo.inc"

using namespace llvm;

BlackfinSubtarget::BlackfinSubtarget(const std::string &TT,
                                     const std::string &CPU,
                                     const std::string &FS)
  : BlackfinGenSubtargetInfo(TT, CPU, FS), sdram(false),
    icplb(false),
    wa_mi_shift(false),
    wa_csync(false),
    wa_specld(false),
    wa_mmr_stall(false),
    wa_lcregs(false),
    wa_hwloop(false),
    wa_ind_call(false),
    wa_killed_mmr(false),
    wa_rets(false)
{
  std::string CPUName = CPU;
  if (CPUName.empty())
    CPUName = "generic";
  // Parse features string.
  ParseSubtargetFeatures(CPUName, FS);
}
