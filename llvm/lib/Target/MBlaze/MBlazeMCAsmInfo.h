//=====-- MBlazeMCAsmInfo.h - MBlaze asm properties -----------*- C++ -*--====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the MBlazeMCAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef MBLAZETARGETASMINFO_H
#define MBLAZETARGETASMINFO_H

#include "llvm/ADT/StringRef.h"
#include "llvm/MC/MCAsmInfo.h"

namespace llvm {
  class Target;
  
  class MBlazeMCAsmInfo : public MCAsmInfo {
  public:
    explicit MBlazeMCAsmInfo(const Target &T, StringRef TT);
  };

} // namespace llvm

#endif
