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

class Target;
class StringRef;

class AMDGPUMCAsmInfo : public MCAsmInfo {
public:
  explicit AMDGPUMCAsmInfo(const Target &T, StringRef &TT);
  const char* getDataASDirective(unsigned int Size, unsigned int AS) const;
  const MCSection* getNonexecutableStackSection(MCContext &CTX) const;
};
} // namespace llvm
#endif // AMDGPUMCASMINFO_H
