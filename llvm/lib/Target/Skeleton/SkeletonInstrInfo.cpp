//===- SkeletonInstrInfo.cpp - Instruction Information ----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This is where you implement methods for the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "SkeletonInstrInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "SkeletonGenInstrInfo.inc"  // Get info from Tablegen
using namespace llvm;

SkeletonInstrInfo::SkeletonInstrInfo()
  : TargetInstrInfo(SkeletonInsts,
                    sizeof(SkeletonInsts)/sizeof(SkeletonInsts[0])){
}
