//===- PowerPCInstrInfo.cpp - PowerPC Instruction Information ---*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file contains the PowerPC implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "PowerPCInstrInfo.h"
#include "PowerPC.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "PowerPCGenInstrInfo.inc"
using namespace llvm;

PowerPCInstrInfo::PowerPCInstrInfo()
  : TargetInstrInfo(PowerPCInsts, sizeof(PowerPCInsts)/sizeof(PowerPCInsts[0])){
}
