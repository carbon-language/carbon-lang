//===- SystemZSubtarget.cpp - SystemZ Subtarget Information -------*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SystemZ specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#include "SystemZSubtarget.h"
#include "SystemZ.h"
#include "SystemZGenSubtarget.inc"
#include "llvm/Target/TargetMachine.h"

using namespace llvm;

SystemZSubtarget::SystemZSubtarget(const TargetMachine &TM, const Module &M,
                                 const std::string &FS) {
  std::string CPU = "generic";

  // Parse features string.
  ParseSubtargetFeatures(FS, CPU);
}
