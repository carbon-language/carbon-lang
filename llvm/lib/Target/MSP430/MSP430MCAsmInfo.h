//=====-- MSP430MCAsmInfo.h - MSP430 asm properties -----------*- C++ -*--====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the MSP430MCAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef MSP430TARGETASMINFO_H
#define MSP430TARGETASMINFO_H

#include "llvm/MC/MCAsmInfo.h"

namespace llvm {
  class Target;
  class StringRef;
  struct MSP430MCAsmInfo : public MCAsmInfo {
    explicit MSP430MCAsmInfo(const Target &T, const StringRef &TT);
  };

} // namespace llvm

#endif
