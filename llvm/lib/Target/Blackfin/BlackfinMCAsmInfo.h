//===-- BlackfinMCAsmInfo.h - Blackfin asm properties ---------*- C++ -*--====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the BlackfinMCAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef BLACKFINTARGETASMINFO_H
#define BLACKFINTARGETASMINFO_H

#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCAsmInfo.h"

namespace llvm {
  class Target;

  struct BlackfinMCAsmInfo : public MCAsmInfo {
    explicit BlackfinMCAsmInfo(const Target &T, StringRef TT);
  };

} // namespace llvm

#endif
