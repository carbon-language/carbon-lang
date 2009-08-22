//=====-- ARMMCAsmInfo.h - ARM asm properties -------------*- C++ -*--====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the ARMMCAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ARMTARGETASMINFO_H
#define LLVM_ARMTARGETASMINFO_H

#include "llvm/MC/MCAsmInfoDarwin.h"

namespace llvm {

  struct ARMDarwinMCAsmInfo : public DarwinMCAsmInfo {
    explicit ARMDarwinMCAsmInfo();
  };

  struct ARMELFMCAsmInfo : public MCAsmInfo {
    explicit ARMELFMCAsmInfo();
  };

} // namespace llvm

#endif
