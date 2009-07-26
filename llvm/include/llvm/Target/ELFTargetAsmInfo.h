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
  class GlobalVariable;
  class Type;

  struct ELFTargetAsmInfo: public TargetAsmInfo {
    explicit ELFTargetAsmInfo(const TargetMachine &TM);

    /// getSectionForMergeableConstant - Given a mergeable constant with the
    /// specified size and relocation information, return a section that it
    /// should be placed in.
    virtual const Section *
    getSectionForMergeableConstant(SectionKind Kind) const;
    
    /// getFlagsForNamedSection - If this target wants to be able to infer
    /// section flags based on the name of the section specified for a global
    /// variable, it can implement this.  This is used on ELF systems so that
    /// ".tbss" gets the TLS bit set etc.
    virtual unsigned getFlagsForNamedSection(const char *Section) const;
    
    const char *getSectionPrefixForUniqueGlobal(SectionKind Kind) const;
    
    virtual const Section* SelectSectionForGlobal(const GlobalValue *GV,
                                                  SectionKind Kind) const;
    virtual std::string printSectionFlags(unsigned flags) const;

    const Section *DataRelSection;
    const Section *DataRelLocalSection;
    const Section *DataRelROSection;
    const Section *DataRelROLocalSection;

    const Section *MergeableConst4Section;
    const Section *MergeableConst8Section;
    const Section *MergeableConst16Section;

  private:
    const Section *MergeableStringSection(const GlobalVariable *GV) const;
  };
}


#endif // LLVM_ELF_TARGET_ASM_INFO_H
