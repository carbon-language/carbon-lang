//===-- BlackfinMCAsmInfo.cpp - Blackfin asm properties -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the declarations of the BlackfinMCAsmInfo properties.
//
//===----------------------------------------------------------------------===//

#include "BlackfinMCAsmInfo.h"

using namespace llvm;

BlackfinMCAsmInfo::BlackfinMCAsmInfo(const Target &T, StringRef TT) {
  GlobalPrefix = "_";
  CommentString = "//";
  HasSetDirective = false;
}
