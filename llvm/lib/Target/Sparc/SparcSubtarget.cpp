//===- SparcSubtarget.cpp - SPARC Subtarget Information -------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the SPARC specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#include "SparcSubtarget.h"
#include "SparcGenSubtarget.inc"
using namespace llvm;

SparcSubtarget::SparcSubtarget(const std::string &TT, const std::string &FS, 
                               bool is64Bit) :
  IsV9(false),
  V8DeprecatedInsts(false),
  IsVIS(false),
  Is64Bit(is64Bit) {
  
  // Determine default and user specified characteristics
  const char *CPU = "v8";
  if (is64Bit) {
    CPU = "v9";
    IsV9 = true;
  }

  // Parse features string.
  ParseSubtargetFeatures(FS, CPU);
}
