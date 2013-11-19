//===-- HexagonTargetAsmInfo.h - Hexagon asm properties --------*- C++ -*--===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the HexagonMCAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef HexagonMCASMINFO_H
#define HexagonMCASMINFO_H

#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCAsmInfoELF.h"

namespace llvm {
  class HexagonMCAsmInfo : public MCAsmInfoELF {
    virtual void anchor();
  public:
    explicit HexagonMCAsmInfo(StringRef TT);
  };

} // namespace llvm

#endif
