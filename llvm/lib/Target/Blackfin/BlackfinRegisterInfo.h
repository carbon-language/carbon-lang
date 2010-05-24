//===- BlackfinRegisterInfo.h - Blackfin Register Information ..-*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Blackfin implementation of the TargetRegisterInfo
// class.
//
//===----------------------------------------------------------------------===//

#ifndef BLACKFINREGISTERINFO_H
#define BLACKFINREGISTERINFO_H

#include "llvm/Target/TargetRegisterInfo.h"
#include "BlackfinGenRegisterInfo.h.inc"

namespace llvm {

  class BlackfinSubtarget;
  class TargetInstrInfo;
  class Type;

  struct BlackfinRegisterInfo : public BlackfinGenRegisterInfo {
    BlackfinSubtarget &Subtarget;
    const TargetInstrInfo &TII;

    BlackfinRegisterInfo(BlackfinSubtarget &st, const TargetInstrInfo &tii);

    /// Code Generation virtual methods...
    const unsigned *getCalleeSavedRegs(const MachineFunction *MF = 0) const;

    const TargetRegisterClass* const*
    getCalleeSavedRegClasses(const MachineFunction *MF = 0) const;

    BitVector getReservedRegs(const MachineFunction &MF) const;

    // getSubReg implemented by tablegen

    const TargetRegisterClass *getPointerRegClass(unsigned Kind = 0) const {
      return &BF::PRegClass;
    }

    const TargetRegisterClass *getPhysicalRegisterRegClass(unsigned reg,
                                                           EVT VT) const;

    bool hasFP(const MachineFunction &MF) const;

    // bool hasReservedCallFrame(MachineFunction &MF) const;

    bool requiresRegisterScavenging(const MachineFunction &MF) const;

    void eliminateCallFramePseudoInstr(MachineFunction &MF,
                                       MachineBasicBlock &MBB,
                                       MachineBasicBlock::iterator I) const;

    unsigned eliminateFrameIndex(MachineBasicBlock::iterator II,
                                 int SPAdj, FrameIndexValue *Value = NULL,
                                 RegScavenger *RS = NULL) const;

    void processFunctionBeforeCalleeSavedScan(MachineFunction &MF,
                                              RegScavenger *RS) const;

    void processFunctionBeforeFrameFinalized(MachineFunction &MF) const;

    void emitPrologue(MachineFunction &MF) const;
    void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;

    unsigned getFrameRegister(const MachineFunction &MF) const;
    unsigned getRARegister() const;

    // Exception handling queries.
    unsigned getEHExceptionRegister() const;
    unsigned getEHHandlerRegister() const;

    int getDwarfRegNum(unsigned RegNum, bool isEH) const;

    // Utility functions
    void adjustRegister(MachineBasicBlock &MBB,
                        MachineBasicBlock::iterator I,
                        DebugLoc DL,
                        unsigned Reg,
                        unsigned ScratchReg,
                        int delta) const;
    void loadConstant(MachineBasicBlock &MBB,
                      MachineBasicBlock::iterator I,
                      DebugLoc DL,
                      unsigned Reg,
                      int value) const;
  };

} // end namespace llvm

#endif
