//===- AlphaSubtarget.cpp - Alpha Subtarget Information ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Andrew Lenharth and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Alpha specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#include "AlphaSubtarget.h"
#include "Alpha.h"
#include "AlphaGenSubtarget.inc"
using namespace llvm;

AlphaSubtarget::AlphaSubtarget(const Module &M, const std::string &FS)
  : HasF2I(false), HasCT(false) {
  std::string CPU = "generic";

  // Parse features string.
  ParseSubtargetFeatures(FS, CPU);
}
