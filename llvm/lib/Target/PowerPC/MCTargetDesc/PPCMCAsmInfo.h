//=====-- PPCMCAsmInfo.h - PPC asm properties -----------------*- C++ -*--====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the MCAsmInfoDarwin class.
//
//===----------------------------------------------------------------------===//

#ifndef PPCTARGETASMINFO_H
#define PPCTARGETASMINFO_H

#include "llvm/MC/MCAsmInfoDarwin.h"

namespace llvm {

  struct PPCMCAsmInfoDarwin : public MCAsmInfoDarwin {
    explicit PPCMCAsmInfoDarwin(bool is64Bit);
  };

  struct PPCLinuxMCAsmInfo : public MCAsmInfo {
    explicit PPCLinuxMCAsmInfo(bool is64Bit);
  };

} // namespace llvm

#endif
