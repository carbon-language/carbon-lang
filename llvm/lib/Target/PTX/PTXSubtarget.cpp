//===- PTXSubtarget.cpp - PTX Subtarget Information ---------------*- C++ -*-=//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the PTX specific subclass of TargetSubtargetInfo.
//
//===----------------------------------------------------------------------===//

#include "PTXSubtarget.h"
#include "PTX.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/TargetRegistry.h"

#define GET_SUBTARGETINFO_TARGET_DESC
#define GET_SUBTARGETINFO_CTOR
#include "PTXGenSubtargetInfo.inc"

using namespace llvm;

PTXSubtarget::PTXSubtarget(const std::string &TT, const std::string &CPU,
                           const std::string &FS, bool is64Bit)
  : PTXGenSubtargetInfo(TT, CPU, FS),
    PTXTarget(PTX_COMPUTE_1_0),
    PTXVersion(PTX_VERSION_2_0),
    SupportsDouble(false),
    SupportsFMA(true),
    Is64Bit(is64Bit) {
  std::string TARGET = CPU;
  if (TARGET.empty())
    TARGET = "generic";
  ParseSubtargetFeatures(TARGET, FS);
}

std::string PTXSubtarget::getTargetString() const {
  switch(PTXTarget) {
    default: llvm_unreachable("Unknown PTX target");
    case PTX_SM_1_0: return "sm_10";
    case PTX_SM_1_1: return "sm_11";
    case PTX_SM_1_2: return "sm_12";
    case PTX_SM_1_3: return "sm_13";
    case PTX_SM_2_0: return "sm_20";
    case PTX_SM_2_1: return "sm_21";
    case PTX_SM_2_2: return "sm_22";
    case PTX_SM_2_3: return "sm_23";
    case PTX_COMPUTE_1_0: return "compute_10";
    case PTX_COMPUTE_1_1: return "compute_11";
    case PTX_COMPUTE_1_2: return "compute_12";
    case PTX_COMPUTE_1_3: return "compute_13";
    case PTX_COMPUTE_2_0: return "compute_20";
  }
}

std::string PTXSubtarget::getPTXVersionString() const {
  switch(PTXVersion) {
    default: llvm_unreachable("Unknown PTX version");
    case PTX_VERSION_2_0: return "2.0";
    case PTX_VERSION_2_1: return "2.1";
    case PTX_VERSION_2_2: return "2.2";
    case PTX_VERSION_2_3: return "2.3";
  }
}
