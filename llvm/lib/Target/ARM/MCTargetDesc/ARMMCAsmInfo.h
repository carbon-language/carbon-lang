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

#include "llvm/MC/MCAsmInfoCOFF.h"
#include "llvm/MC/MCAsmInfoDarwin.h"
#include "llvm/MC/MCAsmInfoELF.h"

namespace llvm {

  class ARMMCAsmInfoDarwin : public MCAsmInfoDarwin {
    void anchor() override;
  public:
    explicit ARMMCAsmInfoDarwin(StringRef TT);
  };

  class ARMELFMCAsmInfo : public MCAsmInfoELF {
    void anchor() override;
  public:
    explicit ARMELFMCAsmInfo(StringRef TT);

    void setUseIntegratedAssembler(bool Value) override;
  };

  class ARMCOFFMCAsmInfoMicrosoft : public MCAsmInfoMicrosoft {
    void anchor();
  public:
    explicit ARMCOFFMCAsmInfoMicrosoft();
  };

  class ARMCOFFMCAsmInfoGNU : public MCAsmInfoGNUCOFF {
    void anchor();
  public:
    explicit ARMCOFFMCAsmInfoGNU();
  };

} // namespace llvm

#endif
