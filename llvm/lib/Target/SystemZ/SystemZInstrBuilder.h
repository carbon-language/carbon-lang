//===-- SystemZInstrBuilder.h - Functions to aid building insts -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file exposes functions that may be used with BuildMI from the
// MachineInstrBuilder.h file to handle SystemZ'isms in a clean way.
//
//===----------------------------------------------------------------------===//

#ifndef SYSTEMZINSTRBUILDER_H
#define SYSTEMZINSTRBUILDER_H

#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/PseudoSourceValue.h"

namespace llvm {

/// Add a BDX memory reference for frame object FI to MIB.
static inline const MachineInstrBuilder &
addFrameReference(const MachineInstrBuilder &MIB, int FI) {
  MachineInstr *MI = MIB;
  MachineFunction &MF = *MI->getParent()->getParent();
  MachineFrameInfo *MFFrame = MF.getFrameInfo();
  const MCInstrDesc &MCID = MI->getDesc();
  unsigned Flags = 0;
  if (MCID.mayLoad())
    Flags |= MachineMemOperand::MOLoad;
  if (MCID.mayStore())
    Flags |= MachineMemOperand::MOStore;
  int64_t Offset = 0;
  MachineMemOperand *MMO =
    MF.getMachineMemOperand(MachinePointerInfo(
                              PseudoSourceValue::getFixedStack(FI), Offset),
                            Flags, MFFrame->getObjectSize(FI),
                            MFFrame->getObjectAlignment(FI));
  return MIB.addFrameIndex(FI).addImm(Offset).addReg(0).addMemOperand(MMO);
}

} // end namespace llvm

#endif
