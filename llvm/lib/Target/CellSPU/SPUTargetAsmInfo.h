//===-- SPUTargetAsmInfo.h - Cell SPU asm properties -----------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the SPUTargetAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef SPUTARGETASMINFO_H
#define SPUTARGETASMINFO_H

#include "llvm/Target/TargetAsmInfo.h"

namespace llvm {
  class Target;
  class StringRef;
  
  struct SPULinuxTargetAsmInfo : public TargetAsmInfo {
    explicit SPULinuxTargetAsmInfo(const Target &T, const StringRef &TT);
  };
} // namespace llvm

#endif /* SPUTARGETASMINFO_H */
