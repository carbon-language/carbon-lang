//=====-- XCoreMCAsmInfo.h - XCore asm properties -------------*- C++ -*--====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the XCoreMCAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef XCORETARGETASMINFO_H
#define XCORETARGETASMINFO_H

#include "llvm/MC/MCAsmInfo.h"

namespace llvm {
  class Target;
  class StringRef;
  class XCoreMCAsmInfo : public MCAsmInfo {
  public:
    explicit XCoreMCAsmInfo(const Target &T, const StringRef &TT);
  };

} // namespace llvm

#endif
