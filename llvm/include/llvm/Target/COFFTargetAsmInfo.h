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
  public:
    virtual void getSectionFlagsAsString(SectionKind Kind,
                                         SmallVectorImpl<char> &Str) const;
    
    virtual const Section *
    SelectSectionForGlobal(const GlobalValue *GV, SectionKind Kind) const;
      
  };
}


#endif // LLVM_ELF_TARGET_ASM_INFO_H
