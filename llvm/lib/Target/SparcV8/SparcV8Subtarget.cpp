//===- SparcV8Subtarget.cpp - SPARC Subtarget Information -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Chris Lattner and is distributed under the
// University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SPARC specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#include "SparcV8Subtarget.h"
#include "SparcV8GenSubtarget.inc"
using namespace llvm;

SparcV8Subtarget::SparcV8Subtarget(const Module &M, const std::string &FS) {
  // Determine default and user specified characteristics
  std::string CPU = "generic";

  // FIXME: autodetect host here!
  
  // Parse features string.
  ParseSubtargetFeatures(FS, CPU);

};