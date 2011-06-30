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

SparcSubtarget::SparcSubtarget(const std::string &TT, const std::string &CPU,
                               const std::string &FS,  bool is64Bit) :
  IsV9(false),
  V8DeprecatedInsts(false),
  IsVIS(false),
  Is64Bit(is64Bit) {
  
  // Determine default and user specified characteristics
  std::string CPUName = CPU;
  if (CPUName.empty()) {
    if (is64Bit)
      CPUName = "v9";
    else
      CPUName = "v8";
  }
  IsV9 = CPUName == "v9";

  // Parse features string.
  ParseSubtargetFeatures(FS, CPUName);
}
