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

namespace llvm {
  class Target;
  class StringRef;
  
  class MipsTargetAsmInfo : public TargetAsmInfo {
  public:
    explicit MipsTargetAsmInfo(const Target &T, const StringRef &TT);
  };

} // namespace llvm

#endif
