//===-- ARMELFWriterInfo.cpp - ELF Writer Info for the ARM backend --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements ELF writer information for the ARM backend.
//
//===----------------------------------------------------------------------===//

#include "ARMELFWriterInfo.h"
#include "ARMRelocations.h"
#include "llvm/Function.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/DataLayout.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/ELF.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
//  Implementation of the ARMELFWriterInfo class
//===----------------------------------------------------------------------===//

ARMELFWriterInfo::ARMELFWriterInfo(TargetMachine &TM)
  : TargetELFWriterInfo(TM.getDataLayout()->getPointerSizeInBits() == 64,
                        TM.getDataLayout()->isLittleEndian()) {
}

ARMELFWriterInfo::~ARMELFWriterInfo() {}

unsigned ARMELFWriterInfo::getRelocationType(unsigned MachineRelTy) const {
  switch (MachineRelTy) {
  case ARM::reloc_arm_absolute:
  case ARM::reloc_arm_relative:
  case ARM::reloc_arm_cp_entry:
  case ARM::reloc_arm_vfp_cp_entry:
  case ARM::reloc_arm_machine_cp_entry:
  case ARM::reloc_arm_jt_base:
  case ARM::reloc_arm_pic_jt:
    llvm_unreachable("unsupported ARM relocation type");

  case ARM::reloc_arm_branch: return ELF::R_ARM_CALL;
  case ARM::reloc_arm_movt:   return ELF::R_ARM_MOVT_ABS;
  case ARM::reloc_arm_movw:   return ELF::R_ARM_MOVW_ABS_NC;
  default:
    llvm_unreachable("unknown ARM relocation type");
  }
}

long int ARMELFWriterInfo::getDefaultAddendForRelTy(unsigned RelTy,
                                                    long int Modifier) const {
  llvm_unreachable("ARMELFWriterInfo::getDefaultAddendForRelTy() not "
                   "implemented");
}

unsigned ARMELFWriterInfo::getRelocationTySize(unsigned RelTy) const {
  llvm_unreachable("ARMELFWriterInfo::getRelocationTySize() not implemented");
}

bool ARMELFWriterInfo::isPCRelativeRel(unsigned RelTy) const {
  llvm_unreachable("ARMELFWriterInfo::isPCRelativeRel() not implemented");
}

unsigned ARMELFWriterInfo::getAbsoluteLabelMachineRelTy() const {
  llvm_unreachable("ARMELFWriterInfo::getAbsoluteLabelMachineRelTy() not "
                   "implemented");
}

long int ARMELFWriterInfo::computeRelocation(unsigned SymOffset,
                                             unsigned RelOffset,
                                             unsigned RelTy) const {
  llvm_unreachable("ARMELFWriterInfo::getAbsoluteLabelMachineRelTy() not "
                   "implemented");
}
