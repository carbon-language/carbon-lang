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
    explicit MipsMCAsmInfo(const Target &T, const StringRef &TT,
                           bool isLittleEndian);
  };
  
  /// Big Endian MAI.
  class MipsBEMCAsmInfo : public MipsMCAsmInfo {
  public:
    MipsBEMCAsmInfo(const Target &T, const StringRef &TT)
      : MipsMCAsmInfo(T, TT, false) {}
  };
  
  /// Little Endian MAI.
  class MipsLEMCAsmInfo : public MipsMCAsmInfo {
  public:
    MipsLEMCAsmInfo(const Target &T, const StringRef &TT)
    : MipsMCAsmInfo(T, TT, true) {}
  };
} // namespace llvm

#endif
