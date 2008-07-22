//=====-- MipsTargetAsmInfo.h - Mips asm properties -----------*- C++ -*--====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the MipsTargetAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef MIPSTARGETASMINFO_H
#define MIPSTARGETASMINFO_H

#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/ELFTargetAsmInfo.h"
#include "llvm/DerivedTypes.h"
#include "llvm/GlobalVariable.h"
#include "llvm/Target/TargetOptions.h"

namespace llvm {

  // Forward declaration.
  class MipsTargetMachine;

  struct MipsTargetAsmInfo : public ELFTargetAsmInfo {
    explicit MipsTargetAsmInfo(const MipsTargetMachine &TM);

    /// SectionKindForGlobal - This hook allows the target to select proper
    /// section kind used for global emission.
    virtual SectionKind::Kind
    SectionKindForGlobal(const GlobalValue *GV) const;

    virtual const Section* SelectSectionForGlobal(const GlobalValue *GV) const;

    private:
      const MipsTargetMachine *MipsTM; 

  };

} // namespace llvm

#endif
