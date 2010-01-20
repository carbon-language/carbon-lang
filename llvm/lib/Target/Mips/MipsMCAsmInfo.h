//=====-- MipsMCAsmInfo.h - Mips asm properties ---------------*- C++ -*--====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the MipsMCAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef MIPSTARGETASMINFO_H
#define MIPSTARGETASMINFO_H

#include "llvm/MC/MCAsmInfo.h"

namespace llvm {
  class Target;
  class StringRef;
  
  class MipsMCAsmInfo : public MCAsmInfo {
  public:
    explicit MipsMCAsmInfo(const Target &T, const StringRef &TT);
  };

} // namespace llvm

#endif
