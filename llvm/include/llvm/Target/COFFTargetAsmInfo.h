//===-- COFFTargetAsmInfo.h - COFF asm properties ---------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_COFF_TARGET_ASM_INFO_H
#define LLVM_COFF_TARGET_ASM_INFO_H

#include "llvm/Target/TargetAsmInfo.h"

namespace llvm {
  class COFFTargetAsmInfo : public TargetAsmInfo {
  protected:
    explicit COFFTargetAsmInfo(const TargetMachine &TM);
      
  };
}


#endif // LLVM_COFF_TARGET_ASM_INFO_H
