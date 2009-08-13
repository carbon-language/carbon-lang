//=====-- ARMTargetAsmInfo.h - ARM asm properties -------------*- C++ -*--====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the ARMTargetAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ARMTARGETASMINFO_H
#define LLVM_ARMTARGETASMINFO_H

#include "llvm/Target/DarwinTargetAsmInfo.h"

namespace llvm {

  struct ARMDarwinTargetAsmInfo : public DarwinTargetAsmInfo {
    explicit ARMDarwinTargetAsmInfo();
  };

  struct ARMELFTargetAsmInfo : public TargetAsmInfo {
    explicit ARMELFTargetAsmInfo();
  };

} // namespace llvm

#endif
