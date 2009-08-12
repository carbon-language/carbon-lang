//====-- SystemZTargetAsmInfo.h - SystemZ asm properties -------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the SystemZTargetAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef SystemZTARGETASMINFO_H
#define SystemZTARGETASMINFO_H

#include "llvm/Target/TargetAsmInfo.h"

namespace llvm {
  class Target;
  class StringRef;

  struct SystemZTargetAsmInfo : public TargetAsmInfo {
    explicit SystemZTargetAsmInfo(const Target &T, const StringRef &TT);
  };

} // namespace llvm

#endif
