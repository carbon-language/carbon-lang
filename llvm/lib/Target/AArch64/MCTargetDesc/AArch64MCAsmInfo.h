//==-- AArch64MCAsmInfo.h - AArch64 asm properties -------------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the AArch64MCAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_AARCH64TARGETASMINFO_H
#define LLVM_AARCH64TARGETASMINFO_H

#include "llvm/MC/MCAsmInfoELF.h"

namespace llvm {

struct AArch64ELFMCAsmInfo : public MCAsmInfoELF {
  explicit AArch64ELFMCAsmInfo(StringRef TT);
private:
  void anchor() override;
};

} // namespace llvm

#endif
