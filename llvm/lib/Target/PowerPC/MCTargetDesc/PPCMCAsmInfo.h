//===-- PPCMCAsmInfo.h - PPC asm properties --------------------*- C++ -*--===//
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
#include "llvm/MC/MCAsmInfoELF.h"

namespace llvm {
class Triple;

  class PPCMCAsmInfoDarwin : public MCAsmInfoDarwin {
    void anchor() override;
  public:
    explicit PPCMCAsmInfoDarwin(bool is64Bit, const Triple&);
  };

  class PPCLinuxMCAsmInfo : public MCAsmInfoELF {
    void anchor() override;
  public:
    explicit PPCLinuxMCAsmInfo(bool is64Bit, const Triple&);
  };

} // namespace llvm

#endif
