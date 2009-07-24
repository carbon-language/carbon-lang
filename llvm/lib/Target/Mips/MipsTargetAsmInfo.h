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

#include "MipsSubtarget.h"
#include "llvm/DerivedTypes.h"
#include "llvm/Target/TargetAsmInfo.h"
#include "llvm/Target/TargetOptions.h"
#include "llvm/Target/ELFTargetAsmInfo.h"

namespace llvm {

  // Forward declaration.
  class GlobalValue;
  class MipsTargetMachine;

  struct MipsTargetAsmInfo : public ELFTargetAsmInfo {
    explicit MipsTargetAsmInfo(const MipsTargetMachine &TM);

    private:
      const MipsSubtarget *Subtarget;
  };

} // namespace llvm

#endif
