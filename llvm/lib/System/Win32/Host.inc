//===- llvm/System/Win32/Host.inc -------------------------------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Win32 Host support.
//
//===----------------------------------------------------------------------===//

#include "Win32.h"
#include <cstdio>
#include <string>

using namespace llvm;

std::string sys::getHostTriple() {
  // FIXME: Adapt to running version.
  return LLVM_HOSTTRIPLE;
}
