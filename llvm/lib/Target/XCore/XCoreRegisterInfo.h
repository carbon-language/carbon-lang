//===-- XCoreRegisterInfo.h - XCore Register Information Impl ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the XCore implementation of the MRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef XCOREREGISTERINFO_H
#define XCOREREGISTERINFO_H

#include "llvm/Target/TargetRegisterInfo.h"

#define GET_REGINFO_HEADER
#include "XCoreGenRegisterInfo.inc"

namespace llvm {

class TargetInstrInfo;

struct XCoreRegisterInfo : public XCoreGenRegisterInfo {
public:
  XCoreRegisterInfo();

  /// Code Generation virtual methods...

  const MCPhysReg *
  getCalleeSavedRegs(const MachineFunction *MF =nullptr) const override;

  BitVector getReservedRegs(const MachineFunction &MF) const override;
  
  bool requiresRegisterScavenging(const MachineFunction &MF) const override;

  bool trackLivenessAfterRegAlloc(const MachineFunction &MF) const override;

  bool useFPForScavengingIndex(const MachineFunction &MF) const override;

  void eliminateFrameIndex(MachineBasicBlock::iterator II,
                           int SPAdj, unsigned FIOperandNum,
                           RegScavenger *RS = nullptr) const override;

  // Debug information queries.
  unsigned getFrameRegister(const MachineFunction &MF) const override;

  //! Return whether to emit frame moves
  static bool needsFrameMoves(const MachineFunction &MF);
};

} // end namespace llvm

#endif
