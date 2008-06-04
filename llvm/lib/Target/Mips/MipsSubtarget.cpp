//===- MipsSubtarget.cpp - Mips Subtarget Information -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file implements the Mips specific subclass of TargetSubtarget.
//
//===----------------------------------------------------------------------===//

#include "MipsSubtarget.h"
#include "Mips.h"
#include "MipsGenSubtarget.inc"
using namespace llvm;

MipsSubtarget::MipsSubtarget(const TargetMachine &TM, const Module &M, 
                             const std::string &FS, bool little) : 
  IsMipsIII(false),
  IsLittle(little)
{
  std::string CPU = "mips1";

  // Parse features string.
  ParseSubtargetFeatures(FS, CPU);
}
