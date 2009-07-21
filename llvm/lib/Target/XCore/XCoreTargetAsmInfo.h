//=====-- XCoreTargetAsmInfo.h - XCore asm properties ---------*- C++ -*--====//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declaration of the XCoreTargetAsmInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef XCORETARGETASMINFO_H
#define XCORETARGETASMINFO_H

#include "llvm/Target/ELFTargetAsmInfo.h"

namespace llvm {

  // Forward declarations.
  class XCoreTargetMachine;
  class XCoreSubtarget;

  class XCoreTargetAsmInfo : public ELFTargetAsmInfo {
  public:
    explicit XCoreTargetAsmInfo(const XCoreTargetMachine &TM);
  };

} // namespace llvm

#endif
