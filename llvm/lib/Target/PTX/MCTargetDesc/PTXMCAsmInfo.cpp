//===-- PTXMCAsmInfo.cpp - PTX asm properties -----------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the PTXMCAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "PTXMCAsmInfo.h"
#include "llvm/ADT/Triple.h"

using namespace llvm;

void PTXMCAsmInfo::anchor() { }

PTXMCAsmInfo::PTXMCAsmInfo(const Target &T, const StringRef &TT) {
  Triple TheTriple(TT);
  if (TheTriple.getArch() == Triple::ptx64)
    PointerSize = 8;

  CommentString = "//";

  PrivateGlobalPrefix = "$L__";

  AllowPeriodsInName = false;

  HasSetDirective = false;

  HasDotTypeDotSizeDirective = false;

  HasSingleParameterDotFile = false;
}
