//===- MSP430InstrInfo.cpp - MSP430 Instruction Information ---------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the MSP430 implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "MSP430.h"
#include "MSP430InstrInfo.h"
#include "MSP430TargetMachine.h"
#include "MSP430GenInstrInfo.inc"
#include "llvm/Function.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"


using namespace llvm;

MSP430InstrInfo::MSP430InstrInfo(MSP430TargetMachine &tm)
  : TargetInstrInfoImpl(MSP430Insts, array_lengthof(MSP430Insts)),
    RI(*this), TM(tm) {}
