//===- SparcV8InstrInfo.cpp - SparcV8 Instruction Information ---*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file contains the SparcV8 implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "SparcV8InstrInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "SparcV8GenInstrInfo.inc"
using namespace llvm;

SparcV8InstrInfo::SparcV8InstrInfo()
  : TargetInstrInfo(SparcV8Insts, sizeof(SparcV8Insts)/sizeof(SparcV8Insts[0])){
}

