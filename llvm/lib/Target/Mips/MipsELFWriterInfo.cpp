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
#include "llvm/Support/ELF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"

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
  if (is64Bit) {
    switch(MachineRelTy) {
    default:
      llvm_unreachable("unknown Mips_64 machine relocation type");
    }
  } else {
    switch(MachineRelTy) {
    case Mips::reloc_mips_pcrel:
      return ELF::R_MIPS_PC16;
    case Mips::reloc_mips_hi:
      return ELF::R_MIPS_HI16;
    case Mips::reloc_mips_lo:
      return ELF::R_MIPS_LO16;
    case Mips::reloc_mips_j_jal:
      return ELF::R_MIPS_26;
    case Mips::reloc_mips_16:
      return ELF::R_MIPS_16;
    case Mips::reloc_mips_32:
      return ELF::R_MIPS_32;
    case Mips::reloc_mips_rel32:
      return ELF::R_MIPS_REL32;
    case Mips::reloc_mips_gprel16:
      return ELF::R_MIPS_GPREL16;
    case Mips::reloc_mips_literal:
      return ELF::R_MIPS_LITERAL;
    case Mips::reloc_mips_got16:
      return ELF::R_MIPS_GOT16;
    case Mips::reloc_mips_call16:
      return ELF::R_MIPS_CALL16;
    case Mips::reloc_mips_gprel32:
      return ELF::R_MIPS_GPREL32;
    case Mips::reloc_mips_shift5:
      return ELF::R_MIPS_SHIFT5;
    case Mips::reloc_mips_shift6:
      return ELF::R_MIPS_SHIFT6;
    case Mips::reloc_mips_64:
      return ELF::R_MIPS_64;
    case Mips::reloc_mips_tlsgd:
      return ELF::R_MIPS_TLS_GD;
    case Mips::reloc_mips_gottprel:
      return ELF::R_MIPS_TLS_GOTTPREL;
    case Mips::reloc_mips_tprel_hi:
      return ELF::R_MIPS_TLS_TPREL_HI16;
    case Mips::reloc_mips_tprel_lo:
      return ELF::R_MIPS_TLS_TPREL_LO16;
    case Mips::reloc_mips_branch_pcrel:
      return ELF::R_MIPS_PC16;
    default:
      llvm_unreachable("unknown Mips machine relocation type");
    }
  }
  return 0;
}

long int MipsELFWriterInfo::getDefaultAddendForRelTy(unsigned RelTy,
                                                     long int Modifier) const {
  if (is64Bit) {
    switch(RelTy) {
    default:
      llvm_unreachable("unknown Mips_64 relocation type");
    }
  } else {
    switch(RelTy) {
    case ELF::R_MIPS_PC16: return Modifier - 4;
    default:
      llvm_unreachable("unknown Mips relocation type");
    }
  }
  return 0;
}

unsigned MipsELFWriterInfo::getRelocationTySize(unsigned RelTy) const {
  if (is64Bit) {
    switch(RelTy) {
    case ELF::R_MIPS_PC16:
    case ELF::R_MIPS_HI16:
    case ELF::R_MIPS_LO16:
    case ELF::R_MIPS_26:
    case ELF::R_MIPS_16:
    case ELF::R_MIPS_32:
    case ELF::R_MIPS_REL32:
    case ELF::R_MIPS_GPREL16:
    case ELF::R_MIPS_LITERAL:
    case ELF::R_MIPS_GOT16:
    case ELF::R_MIPS_CALL16:
    case ELF::R_MIPS_GPREL32:
    case ELF::R_MIPS_SHIFT5:
    case ELF::R_MIPS_SHIFT6:
      return 32;
    case ELF::R_MIPS_64:
      return 64;
    default:
      llvm_unreachable("unknown Mips_64 relocation type");
    }
  } else {
    switch(RelTy) {
    case ELF::R_MIPS_PC16:
    case ELF::R_MIPS_HI16:
    case ELF::R_MIPS_LO16:
    case ELF::R_MIPS_26:
    case ELF::R_MIPS_16:
    case ELF::R_MIPS_32:
    case ELF::R_MIPS_REL32:
    case ELF::R_MIPS_GPREL16:
    case ELF::R_MIPS_LITERAL:
    case ELF::R_MIPS_GOT16:
    case ELF::R_MIPS_CALL16:
    case ELF::R_MIPS_GPREL32:
    case ELF::R_MIPS_SHIFT5:
    case ELF::R_MIPS_SHIFT6:
      return 32;
    default:
      llvm_unreachable("unknown Mips relocation type");
    }
  }
  return 0;
}

bool MipsELFWriterInfo::isPCRelativeRel(unsigned RelTy) const {
  if (is64Bit) {
    switch(RelTy) {
    case ELF::R_MIPS_PC16:
      return true;
    case ELF::R_MIPS_HI16:
    case ELF::R_MIPS_LO16:
    case ELF::R_MIPS_26:
    case ELF::R_MIPS_16:
    case ELF::R_MIPS_32:
    case ELF::R_MIPS_REL32:
    case ELF::R_MIPS_GPREL16:
    case ELF::R_MIPS_LITERAL:
    case ELF::R_MIPS_GOT16:
    case ELF::R_MIPS_CALL16:
    case ELF::R_MIPS_GPREL32:
    case ELF::R_MIPS_SHIFT5:
    case ELF::R_MIPS_SHIFT6:
    case ELF::R_MIPS_64:
      return false;
    default:
      llvm_unreachable("unknown Mips_64 relocation type");
    }
  } else {
    switch(RelTy) {
    case ELF::R_MIPS_PC16:
      return true;
    case ELF::R_MIPS_HI16:
    case ELF::R_MIPS_LO16:
    case ELF::R_MIPS_26:
    case ELF::R_MIPS_16:
    case ELF::R_MIPS_32:
    case ELF::R_MIPS_REL32:
    case ELF::R_MIPS_GPREL16:
    case ELF::R_MIPS_LITERAL:
    case ELF::R_MIPS_GOT16:
    case ELF::R_MIPS_CALL16:
    case ELF::R_MIPS_GPREL32:
    case ELF::R_MIPS_SHIFT5:
    case ELF::R_MIPS_SHIFT6:
      return false;
    default:
      llvm_unreachable("unknown Mips relocation type");
    }
  }
  return 0;
}

unsigned MipsELFWriterInfo::getAbsoluteLabelMachineRelTy() const {
  assert("getAbsoluteLabelMachineRelTy unknown for this relocation type");
  return 0;
}

long int MipsELFWriterInfo::computeRelocation(unsigned SymOffset,
                                              unsigned RelOffset,
                                              unsigned RelTy) const {
  if (RelTy == ELF::R_MIPS_PC16)
    return SymOffset - (RelOffset + 4);
  else
    assert("computeRelocation unknown for this relocation type");

  return 0;
}
