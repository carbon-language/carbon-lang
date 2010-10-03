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
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
//  Implementation of the ARMELFWriterInfo class
//===----------------------------------------------------------------------===//

ARMELFWriterInfo::ARMELFWriterInfo(TargetMachine &TM)
  : TargetELFWriterInfo(TM.getTargetData()->getPointerSizeInBits() == 64,
                        TM.getTargetData()->isLittleEndian()) {
  // silently OK construction
}

ARMELFWriterInfo::~ARMELFWriterInfo() {}

unsigned ARMELFWriterInfo::getRelocationType(unsigned MachineRelTy) const {
  assert(0 && "ARMELFWriterInfo::getRelocationType() not implemented");
  return 0;
}

long int ARMELFWriterInfo::getDefaultAddendForRelTy(unsigned RelTy,
                                                    long int Modifier) const {
  assert(0 && "ARMELFWriterInfo::getDefaultAddendForRelTy() not implemented");
  return 0;
}

unsigned ARMELFWriterInfo::getRelocationTySize(unsigned RelTy) const {
  assert(0 && "ARMELFWriterInfo::getRelocationTySize() not implemented");
  return 0;
}

bool ARMELFWriterInfo::isPCRelativeRel(unsigned RelTy) const {
  assert(0 && "ARMELFWriterInfo::isPCRelativeRel() not implemented");
  return 1;
}

unsigned ARMELFWriterInfo::getAbsoluteLabelMachineRelTy() const {
  assert(0 &&
         "ARMELFWriterInfo::getAbsoluteLabelMachineRelTy() not implemented");
  return 0;
}

long int ARMELFWriterInfo::computeRelocation(unsigned SymOffset,
                                             unsigned RelOffset,
                                             unsigned RelTy) const {
  assert(0 &&
         "ARMELFWriterInfo::getAbsoluteLabelMachineRelTy() not implemented");
  return 0;
}
