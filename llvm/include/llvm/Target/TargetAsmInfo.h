//===-- llvm/Target/TargetAsmInfo.h -----------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Interface to provide the information necessary for producing assembly files.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGET_TARGETASMINFO_H
#define LLVM_TARGET_TARGETASMINFO_H

#include "llvm/Target/TargetLoweringObjectFile.h"

namespace llvm {
  class TargetMachine;
  class TargetLoweringObjectFile;

class TargetAsmInfo {
  const TargetLoweringObjectFile *TLOF;

public:
  explicit TargetAsmInfo(const TargetMachine &TM);

  unsigned getFDEEncoding(bool CFI) const {
    return TLOF->getFDEEncoding(CFI);
  }
};

}
#endif
