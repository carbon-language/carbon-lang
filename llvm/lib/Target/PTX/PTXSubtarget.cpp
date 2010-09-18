//===- PTXSubtarget.cpp - PTX Subtarget Information ---------------*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the PTX specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#include "PTXSubtarget.h"

using namespace llvm;

PTXSubtarget::PTXSubtarget(const std::string &TT, const std::string &FS) {
  std::string TARGET = "sm_20";
  // TODO: call ParseSubtargetFeatures(FS, TARGET);
}

#include "PTXGenSubtarget.inc"
