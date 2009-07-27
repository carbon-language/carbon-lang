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
  class GlobalValue;

  struct ELFTargetAsmInfo : public TargetAsmInfo {
    ELFTargetAsmInfo(const TargetMachine &TM);

    /// getSectionForMergeableConstant - Given a mergeable constant with the
    /// specified size and relocation information, return a section that it
    /// should be placed in.
    virtual const Section *
    getSectionForMergeableConstant(SectionKind Kind) const;
    
    virtual SectionKind::Kind getKindForNamedSection(const char *Section,
                                                     SectionKind::Kind K) const;
    void getSectionFlagsAsString(SectionKind Kind,
                                 SmallVectorImpl<char> &Str) const;
    
    const char *getSectionPrefixForUniqueGlobal(SectionKind Kind) const;
    
    virtual const Section* SelectSectionForGlobal(const GlobalValue *GV,
                                                  SectionKind Kind) const;
    
    const Section *DataRelSection;
    const Section *DataRelLocalSection;
    const Section *DataRelROSection;
    const Section *DataRelROLocalSection;

    const Section *MergeableConst4Section;
    const Section *MergeableConst8Section;
    const Section *MergeableConst16Section;
  };
}


#endif // LLVM_ELF_TARGET_ASM_INFO_H
