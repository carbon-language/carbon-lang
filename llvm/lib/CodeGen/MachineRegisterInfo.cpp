//===-- MachineRegisterInfo.cpp -------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// Implementation of the MachineRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineRegisterInfo.h"
using namespace llvm;

MachineRegisterInfo::MachineRegisterInfo(const MRegisterInfo &MRI) {
  VRegInfo.reserve(256);
  UsedPhysRegs.resize(MRI.getNumRegs());
}
