//===-- MCTargetDesc/AMDGPUMCAsmInfo.h - AMDGPU MCAsm Interface -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
/// \file
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_R600_MCTARGETDESC_AMDGPUMCASMINFO_H
#define LLVM_LIB_TARGET_R600_MCTARGETDESC_AMDGPUMCASMINFO_H

#include "llvm/MC/MCAsmInfo.h"
namespace llvm {

class StringRef;

class AMDGPUMCAsmInfo : public MCAsmInfo {
public:
  explicit AMDGPUMCAsmInfo(StringRef &TT);
  const MCSection* getNonexecutableStackSection(MCContext &CTX) const override;
};
} // namespace llvm
#endif
