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

namespace llvm {

  // Forward declaration.
  class MipsTargetMachine;

  struct MipsTargetAsmInfo : public ELFTargetAsmInfo {
    explicit MipsTargetAsmInfo(const MipsTargetMachine &TM);
  };

} // namespace llvm

#endif
