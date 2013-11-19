//===-- SIMachineFunctionInfo.cpp - SI Machine Function Info -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
/// \file
//===----------------------------------------------------------------------===//


#include "SIMachineFunctionInfo.h"

using namespace llvm;


// Pin the vtable to this file.
void SIMachineFunctionInfo::anchor() {}

SIMachineFunctionInfo::SIMachineFunctionInfo(const MachineFunction &MF)
  : AMDGPUMachineFunction(MF),
    PSInputAddr(0) { }
