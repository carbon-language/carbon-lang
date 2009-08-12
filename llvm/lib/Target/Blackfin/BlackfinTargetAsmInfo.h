//===-- BlackfinTargetAsmInfo.h - Blackfin asm properties -----*- C++ -*--====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the BlackfinTargetAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef BLACKFINTARGETASMINFO_H
#define BLACKFINTARGETASMINFO_H

#include "llvm/Target/TargetAsmInfo.h"

namespace llvm {
  class Target;
  class StringRef;

  struct BlackfinTargetAsmInfo : public TargetAsmInfo {
    explicit BlackfinTargetAsmInfo(const Target &T, const StringRef &TT);
  };

} // namespace llvm

#endif
