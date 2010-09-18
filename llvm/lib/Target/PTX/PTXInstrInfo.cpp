//===- PTXInstrInfo.cpp - PTX Instruction Information ---------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the PTX implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "PTXInstrInfo.h"

using namespace llvm;

#include "PTXGenInstrInfo.inc"

PTXInstrInfo::PTXInstrInfo(PTXTargetMachine &_TM)
  : TargetInstrInfoImpl(PTXInsts, array_lengthof(PTXInsts)),
    RI(_TM, *this), TM(_TM) {}
