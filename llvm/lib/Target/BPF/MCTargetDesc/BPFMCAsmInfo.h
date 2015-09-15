//===-- BPFMCAsmInfo.h - BPF asm properties -------------------*- C++ -*--====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the BPFMCAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_BPF_MCTARGETDESC_BPFMCASMINFO_H
#define LLVM_LIB_TARGET_BPF_MCTARGETDESC_BPFMCASMINFO_H

#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/ADT/TargetTuple.h"

namespace llvm {
class Target;

class BPFMCAsmInfo : public MCAsmInfo {
public:
  explicit BPFMCAsmInfo(const TargetTuple &TT) {
    if (TT.getArch() == TargetTuple::bpfeb)
      IsLittleEndian = false;

    PrivateGlobalPrefix = ".L";
    WeakRefDirective = "\t.weak\t";

    UsesELFSectionDirectiveForBSS = true;
    HasSingleParameterDotFile = false;
    HasDotTypeDotSizeDirective = false;

    SupportsDebugInformation = true;
  }
};
}

#endif
