//=====-- MipsTargetAsmInfo.h - Mips asm properties -----------*- C++ -*--====//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Bruno Cardoso Lopes and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the MipsTargetAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef MIPSTARGETASMINFO_H
#define MIPSTARGETASMINFO_H

#include "llvm/Target/TargetAsmInfo.h"

namespace llvm {

  // Forward declaration.
  class MipsTargetMachine;

  struct MipsTargetAsmInfo : public TargetAsmInfo {
    explicit MipsTargetAsmInfo(const MipsTargetMachine &TM);
  };

} // namespace llvm

#endif
