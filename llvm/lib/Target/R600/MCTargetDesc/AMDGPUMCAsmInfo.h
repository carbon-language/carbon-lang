//===-- MCTargetDesc/AMDGPUMCAsmInfo.h - AMDGPU MCAsm Interface  ----------===//
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

#ifndef AMDGPUMCASMINFO_H
#define AMDGPUMCASMINFO_H

#include "llvm/MC/MCAsmInfo.h"
namespace llvm {

class StringRef;

class AMDGPUMCAsmInfo : public MCAsmInfo {
public:
  explicit AMDGPUMCAsmInfo(StringRef &TT);
  const MCSection* getNonexecutableStackSection(MCContext &CTX) const;
};
} // namespace llvm
#endif // AMDGPUMCASMINFO_H
