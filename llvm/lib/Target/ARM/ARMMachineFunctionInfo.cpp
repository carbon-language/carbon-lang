//===-- ARMMachineFuctionInfo.cpp - ARM machine function info -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#include "ARMMachineFunctionInfo.h"

using namespace llvm;

void ARMFunctionInfo::anchor() { }

ARMFunctionInfo::ARMFunctionInfo(MachineFunction &MF)
    : isThumb(MF.getTarget().getSubtarget<ARMSubtarget>().isThumb()),
      hasThumb2(MF.getTarget().getSubtarget<ARMSubtarget>().hasThumb2()),
      StByValParamsPadding(0), ArgRegsSaveSize(0), HasStackFrame(false),
      RestoreSPFromFP(false), LRSpilledForFarJump(false),
      FramePtrSpillOffset(0), GPRCS1Offset(0), GPRCS2Offset(0), DPRCSOffset(0),
      GPRCS1Size(0), GPRCS2Size(0), DPRCSSize(0), JumpTableUId(0),
      PICLabelUId(0), VarArgsFrameIndex(0), HasITBlocks(false),
      GlobalBaseReg(0) {}
