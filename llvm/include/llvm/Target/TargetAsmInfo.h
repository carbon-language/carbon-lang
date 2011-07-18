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
#include "llvm/Target/TargetRegisterInfo.h"

namespace llvm {
  template <typename T> class ArrayRef;
  class MCSection;
  class TargetMachine;
  class TargetLoweringObjectFile;

class TargetAsmInfo {
  const TargetLoweringObjectFile *TLOF;

public:
  explicit TargetAsmInfo(const TargetMachine &TM);

  const MCSection *getDwarfLineSection() const {
    return TLOF->getDwarfLineSection();
  }

  const MCSection *getEHFrameSection() const {
    return TLOF->getEHFrameSection();
  }

  const MCSection *getCompactUnwindSection() const {
    return TLOF->getCompactUnwindSection();
  }

  const MCSection *getDwarfFrameSection() const {
    return TLOF->getDwarfFrameSection();
  }

  const MCSection *getWin64EHFuncTableSection(StringRef Suffix) const {
    return TLOF->getWin64EHFuncTableSection(Suffix);
  }

  const MCSection *getWin64EHTableSection(StringRef Suffix) const {
    return TLOF->getWin64EHTableSection(Suffix);
  }

  unsigned getFDEEncoding(bool CFI) const {
    return TLOF->getFDEEncoding(CFI);
  }

  bool isFunctionEHFrameSymbolPrivate() const {
    return TLOF->isFunctionEHFrameSymbolPrivate();
  }
};

}
#endif
