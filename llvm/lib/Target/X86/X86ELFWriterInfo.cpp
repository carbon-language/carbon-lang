//===-- X86ELFWriterInfo.cpp - ELF Writer Info for the X86 backend --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements ELF writer information for the X86 backend.
//
//===----------------------------------------------------------------------===//

#include "X86ELFWriterInfo.h"
#include "X86Relocations.h"
#include "llvm/Function.h"
#include "llvm/Support/ELF.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Target/TargetData.h"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

//===----------------------------------------------------------------------===//
//  Implementation of the X86ELFWriterInfo class
//===----------------------------------------------------------------------===//

X86ELFWriterInfo::X86ELFWriterInfo(bool is64Bit_, bool isLittleEndian_)
  : TargetELFWriterInfo(is64Bit_, isLittleEndian_) {
    EMachine = is64Bit ? EM_X86_64 : EM_386;
  }

X86ELFWriterInfo::~X86ELFWriterInfo() {}

unsigned X86ELFWriterInfo::getRelocationType(unsigned MachineRelTy) const {
  if (is64Bit) {
    switch(MachineRelTy) {
    case X86::reloc_pcrel_word:
      return ELF::R_X86_64_PC32;
    case X86::reloc_absolute_word:
      return ELF::R_X86_64_32;
    case X86::reloc_absolute_word_sext:
      return ELF::R_X86_64_32S;
    case X86::reloc_absolute_dword:
      return ELF::R_X86_64_64;
    case X86::reloc_picrel_word:
    default:
      llvm_unreachable("unknown x86_64 machine relocation type");
    }
  } else {
    switch(MachineRelTy) {
    case X86::reloc_pcrel_word:
      return ELF::R_386_PC32;
    case X86::reloc_absolute_word:
      return ELF::R_386_32;
    case X86::reloc_absolute_word_sext:
    case X86::reloc_absolute_dword:
    case X86::reloc_picrel_word:
    default:
      llvm_unreachable("unknown x86 machine relocation type");
    }
  }
}

long int X86ELFWriterInfo::getDefaultAddendForRelTy(unsigned RelTy,
                                                    long int Modifier) const {
  if (is64Bit) {
    switch(RelTy) {
    case ELF::R_X86_64_PC32: return Modifier - 4;
    case ELF::R_X86_64_32:
    case ELF::R_X86_64_32S:
    case ELF::R_X86_64_64:
      return Modifier;
    default:
      llvm_unreachable("unknown x86_64 relocation type");
    }
  } else {
    switch(RelTy) {
    case ELF::R_386_PC32: return Modifier - 4;
    case ELF::R_386_32: return Modifier;
    default:
      llvm_unreachable("unknown x86 relocation type");
    }
  }
}

unsigned X86ELFWriterInfo::getRelocationTySize(unsigned RelTy) const {
  if (is64Bit) {
    switch(RelTy) {
    case ELF::R_X86_64_PC32:
    case ELF::R_X86_64_32:
    case ELF::R_X86_64_32S:
        return 32;
    case ELF::R_X86_64_64:
        return 64;
    default:
      llvm_unreachable("unknown x86_64 relocation type");
    }
  } else {
    switch(RelTy) {
    case ELF::R_386_PC32:
    case ELF::R_386_32:
        return 32;
    default:
      llvm_unreachable("unknown x86 relocation type");
    }
  }
}

bool X86ELFWriterInfo::isPCRelativeRel(unsigned RelTy) const {
  if (is64Bit) {
    switch(RelTy) {
    case ELF::R_X86_64_PC32:
        return true;
    case ELF::R_X86_64_32:
    case ELF::R_X86_64_32S:
    case ELF::R_X86_64_64:
        return false;
    default:
      llvm_unreachable("unknown x86_64 relocation type");
    }
  } else {
    switch(RelTy) {
    case ELF::R_386_PC32:
        return true;
    case ELF::R_386_32:
        return false;
    default:
      llvm_unreachable("unknown x86 relocation type");
    }
  }
}

unsigned X86ELFWriterInfo::getAbsoluteLabelMachineRelTy() const {
  return is64Bit ?
    X86::reloc_absolute_dword : X86::reloc_absolute_word;
}

long int X86ELFWriterInfo::computeRelocation(unsigned SymOffset,
                                             unsigned RelOffset,
                                             unsigned RelTy) const {

  if (RelTy == ELF::R_X86_64_PC32 || RelTy == ELF::R_386_PC32)
    return SymOffset - (RelOffset + 4);

  llvm_unreachable("computeRelocation unknown for this relocation type");
}
