//===- BlackfinSubtarget.cpp - BLACKFIN Subtarget Information -------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the blackfin specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#include "BlackfinSubtarget.h"
#include "BlackfinGenSubtarget.inc"

using namespace llvm;

BlackfinSubtarget::BlackfinSubtarget(const TargetMachine &TM,
                                     const Module &M,
                                     const std::string &FS) {
  std::string CPU = "generic";

  // Parse features string.
  ParseSubtargetFeatures(FS, CPU);
}
