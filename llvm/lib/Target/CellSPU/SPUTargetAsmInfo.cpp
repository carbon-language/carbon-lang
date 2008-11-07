//===-- SPUTargetAsmInfo.cpp - Cell SPU asm properties ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the SPUTargetAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "SPUTargetAsmInfo.h"
#include "SPUTargetMachine.h"
#include "llvm/Function.h"
#include "llvm/Support/Compiler.h"

using namespace llvm;

SPULinuxTargetAsmInfo::SPULinuxTargetAsmInfo(const SPUTargetMachine &TM) :
    SPUTargetAsmInfo<ELFTargetAsmInfo>(TM) {
  PCSymbol = ".";
  CommentString = "#";
  GlobalPrefix = "";
  PrivateGlobalPrefix = ".L";
}

/// PreferredEHDataFormat - This hook allows the target to select data
/// format used for encoding pointers in exception handling data. Reason is
/// 0 for data, 1 for code labels, 2 for function pointers. Global is true
/// if the symbol can be relocated.
unsigned
SPULinuxTargetAsmInfo::PreferredEHDataFormat(DwarfEncoding::Target Reason,
                                             bool Global) const {
  // We really need to write something here.
  return TargetAsmInfo::PreferredEHDataFormat(Reason, Global);
}

// Instantiate default implementation.
TEMPLATE_INSTANTIATION(class SPUTargetAsmInfo<TargetAsmInfo>);
