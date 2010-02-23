//===- MBlazeSubtarget.cpp - MBlaze Subtarget Information -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the MBlaze specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#include "MBlazeSubtarget.h"
#include "MBlaze.h"
#include "MBlazeGenSubtarget.inc"
#include "llvm/Support/CommandLine.h"
using namespace llvm;

MBlazeSubtarget::MBlazeSubtarget(const std::string &TT, const std::string &FS):
  HasPipe3(false), HasBarrel(false), HasDiv(false), HasMul(false),
  HasFSL(false), HasEFSL(false), HasMSRSet(false), HasException(false),
  HasPatCmp(false), HasFPU(false), HasESR(false), HasPVR(false),
  HasMul64(false), HasSqrt(false), HasMMU(false)
{
  std::string CPU = "v400";
  MBlazeArchVersion = V400;

  // Parse features string.
  ParseSubtargetFeatures(FS, CPU);
}
