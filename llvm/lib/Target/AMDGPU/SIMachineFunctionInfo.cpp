//===-- SIMachineFunctionInfo.cpp - SI Machine Function Info -------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//


#include "SIMachineFunctionInfo.h"

using namespace llvm;

SIMachineFunctionInfo::SIMachineFunctionInfo(const MachineFunction &MF)
  : MachineFunctionInfo(),
    spi_ps_input_addr(0)
  { }
