//===-- MBlazeELFWriterInfo.cpp - ELF Writer Info for the MBlaze backend --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements ELF writer information for the MBlaze backend.
//
//===----------------------------------------------------------------------===//

#include "MBlazeELFWriterInfo.h"
#include "MBlazeRelocations.h"
#include "llvm/Function.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
//  Implementation of the MBlazeELFWriterInfo class
//===----------------------------------------------------------------------===//

MBlazeELFWriterInfo::MBlazeELFWriterInfo(TargetMachine &TM)
  : TargetELFWriterInfo(TM.getTargetData()->getPointerSizeInBits() == 64,
                        TM.getTargetData()->isLittleEndian()) {
}

MBlazeELFWriterInfo::~MBlazeELFWriterInfo() {}

unsigned MBlazeELFWriterInfo::getRelocationType(unsigned MachineRelTy) const {
  switch(MachineRelTy) {
  case MBlaze::reloc_pcrel_word:
    return R_MICROBLAZE_64_PCREL;
  case MBlaze::reloc_absolute_word:
    return R_MICROBLAZE_NONE;
  default:
    llvm_unreachable("unknown mblaze machine relocation type");
  }
  return 0;
}

long int MBlazeELFWriterInfo::getDefaultAddendForRelTy(unsigned RelTy,
                                                    long int Modifier) const {
  switch(RelTy) {
  case R_MICROBLAZE_32_PCREL:
    return Modifier - 4;
  case R_MICROBLAZE_32:
    return Modifier;
  default:
    llvm_unreachable("unknown mblaze relocation type");
  }
  return 0;
}

unsigned MBlazeELFWriterInfo::getRelocationTySize(unsigned RelTy) const {
  // FIXME: Most of these sizes are guesses based on the name
  switch(RelTy) {
  case R_MICROBLAZE_32:
  case R_MICROBLAZE_32_PCREL:
  case R_MICROBLAZE_32_PCREL_LO:
  case R_MICROBLAZE_32_LO:
  case R_MICROBLAZE_SRO32:
  case R_MICROBLAZE_SRW32:
  case R_MICROBLAZE_32_SYM_OP_SYM:
  case R_MICROBLAZE_GOTOFF_32:
    return 32;

  case R_MICROBLAZE_64_PCREL:
  case R_MICROBLAZE_64:
  case R_MICROBLAZE_GOTPC_64:
  case R_MICROBLAZE_GOT_64:
  case R_MICROBLAZE_PLT_64:
  case R_MICROBLAZE_GOTOFF_64:
    return 64;
  }

  return 0;
}

bool MBlazeELFWriterInfo::isPCRelativeRel(unsigned RelTy) const {
  // FIXME: Most of these are guesses based on the name
  switch(RelTy) {
  case R_MICROBLAZE_32_PCREL:
  case R_MICROBLAZE_64_PCREL:
  case R_MICROBLAZE_32_PCREL_LO:
  case R_MICROBLAZE_GOTPC_64:
    return true;
  }

  return false;
}

unsigned MBlazeELFWriterInfo::getAbsoluteLabelMachineRelTy() const {
  return MBlaze::reloc_absolute_word;
}

long int MBlazeELFWriterInfo::computeRelocation(unsigned SymOffset,
                                                unsigned RelOffset,
                                                unsigned RelTy) const {
  if (RelTy == R_MICROBLAZE_32_PCREL || R_MICROBLAZE_64_PCREL)
    return SymOffset - (RelOffset + 4);
  else
    assert("computeRelocation unknown for this relocation type");

  return 0;
}
