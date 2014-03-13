//===-- ARMMCAsmInfo.h - ARM asm properties --------------------*- C++ -*--===//
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
#include "llvm/MC/MCAsmInfoELF.h"

namespace llvm {

  class ARMMCAsmInfoDarwin : public MCAsmInfoDarwin {
    void anchor() override;
  public:
    explicit ARMMCAsmInfoDarwin();
  };

  class ARMELFMCAsmInfo : public MCAsmInfoELF {
    void anchor() override;
  public:
    explicit ARMELFMCAsmInfo();

    void setUseIntegratedAssembler(bool Value) override;
  };

} // namespace llvm

#endif
