//===---- ELFTargetAsmInfo.h - ELF asm properties ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines target asm properties related what form asm statements
// should take in general on ELF-based targets
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ELF_TARGET_ASM_INFO_H
#define LLVM_ELF_TARGET_ASM_INFO_H

#include "llvm/Target/TargetAsmInfo.h"

namespace llvm {

  struct ELFTargetAsmInfo : public TargetAsmInfo {
    ELFTargetAsmInfo(const TargetMachine &TM);
  };
}


#endif // LLVM_ELF_TARGET_ASM_INFO_H
