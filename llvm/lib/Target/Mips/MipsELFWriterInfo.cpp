//===-- MipsELFWriterInfo.cpp - ELF Writer Info for the Mips backend ------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements ELF writer information for the Mips backend.
//
//===----------------------------------------------------------------------===//

#include "MipsELFWriterInfo.h"
#include "MipsRelocations.h"
#include "llvm/Function.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Support/ELF.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
//  Implementation of the MipsELFWriterInfo class
//===----------------------------------------------------------------------===//

MipsELFWriterInfo::MipsELFWriterInfo(bool is64Bit_, bool isLittleEndian_)
  : TargetELFWriterInfo(is64Bit_, isLittleEndian_) {
  EMachine = EM_MIPS;
}

MipsELFWriterInfo::~MipsELFWriterInfo() {}

unsigned MipsELFWriterInfo::getRelocationType(unsigned MachineRelTy) const {
  switch(MachineRelTy) {
  case Mips::reloc_mips_pc16:
    return ELF::R_MIPS_GOT16;
  case Mips::reloc_mips_hi:
    return ELF::R_MIPS_HI16;
  case Mips::reloc_mips_lo:
    return ELF::R_MIPS_LO16;
  case Mips::reloc_mips_26:
    return ELF::R_MIPS_26;
  default:
    llvm_unreachable("unknown Mips machine relocation type");
  }
}

long int MipsELFWriterInfo::getDefaultAddendForRelTy(unsigned RelTy,
                                                     long int Modifier) const {
  switch(RelTy) {
  case ELF::R_MIPS_26: return Modifier;
  default:
    llvm_unreachable("unknown Mips relocation type");
  }
}

unsigned MipsELFWriterInfo::getRelocationTySize(unsigned RelTy) const {
  switch(RelTy) {
  case ELF::R_MIPS_GOT16:
  case ELF::R_MIPS_26:
      return 32;
  default:
    llvm_unreachable("unknown Mips relocation type");
  }
}

bool MipsELFWriterInfo::isPCRelativeRel(unsigned RelTy) const {
  switch(RelTy) {
  case ELF::R_MIPS_GOT16:
      return true;
  case ELF::R_MIPS_26:
      return false;
  default:
    llvm_unreachable("unknown Mips relocation type");
  }
}

unsigned MipsELFWriterInfo::getAbsoluteLabelMachineRelTy() const {
  return Mips::reloc_mips_26;
}

long int MipsELFWriterInfo::computeRelocation(unsigned SymOffset,
                                              unsigned RelOffset,
                                              unsigned RelTy) const {

  if (RelTy == ELF::R_MIPS_GOT16)
    return SymOffset - (RelOffset + 4);

  llvm_unreachable("computeRelocation unknown for this relocation type");
}
