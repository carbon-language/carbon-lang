//====-- SystemZMCAsmInfo.h - SystemZ asm properties -----------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_SYSTEMZ_MCTARGETDESC_SYSTEMZMCASMINFO_H
#define LLVM_LIB_TARGET_SYSTEMZ_MCTARGETDESC_SYSTEMZMCASMINFO_H

#include "llvm/MC/MCAsmInfoELF.h"
#include "llvm/Support/Compiler.h"

namespace llvm {
class TargetTuple;

class SystemZMCAsmInfo : public MCAsmInfoELF {
public:
  explicit SystemZMCAsmInfo(const TargetTuple &TT);
};

} // end namespace llvm

#endif
