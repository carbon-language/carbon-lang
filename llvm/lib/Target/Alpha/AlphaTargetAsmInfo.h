//=====-- AlphaTargetAsmInfo.h - Alpha asm properties ---------*- C++ -*--====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the AlphaTargetAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef ALPHATARGETASMINFO_H
#define ALPHATARGETASMINFO_H

#include "llvm/Target/TargetAsmInfo.h"

namespace llvm {
  class Target;
  class StringRef;

  struct AlphaTargetAsmInfo : public TargetAsmInfo {
    explicit AlphaTargetAsmInfo(const Target &T, const StringRef &TT);
  };

} // namespace llvm

#endif
