//===---- DarwinTargetAsmInfo.h - Darwin asm properties ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file defines target asm properties related what form asm statements
// should take in general on Darwin-based targets
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DARWIN_TARGET_ASM_INFO_H
#define LLVM_DARWIN_TARGET_ASM_INFO_H

#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/TargetMachine.h"

namespace llvm {
  class GlobalValue;
  class GlobalVariable;

  struct DarwinTargetAsmInfo: public virtual TargetAsmInfo {
    const Section* TextCoalSection;
    const Section* ConstDataCoalSection;
    const Section* ConstDataSection;
    const Section* DataCoalSection;

    explicit DarwinTargetAsmInfo(const TargetMachine &TM);
    virtual const Section* SelectSectionForGlobal(const GlobalValue *GV) const;
    virtual std::string UniqueSectionForGlobal(const GlobalValue* GV,
                                               SectionKind::Kind kind) const;
    const Section* MergeableConstSection(const GlobalVariable *GV) const;
    const Section* MergeableStringSection(const GlobalVariable *GV) const;
  protected:
    const TargetMachine* DTM;
  };
}


#endif // LLVM_DARWIN_TARGET_ASM_INFO_H
