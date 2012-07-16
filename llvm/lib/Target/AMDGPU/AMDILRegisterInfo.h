//===- AMDILRegisterInfo.h - AMDIL Register Information Impl ----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==-----------------------------------------------------------------------===//
//
// This file contains the AMDIL implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef AMDILREGISTERINFO_H_
#define AMDILREGISTERINFO_H_

#include "llvm/Target/TargetRegisterInfo.h"

#define GET_REGINFO_HEADER
#include "AMDGPUGenRegisterInfo.inc"
// See header file for explanation

namespace llvm
{

  class TargetInstrInfo;
  class Type;

  /// DWARFFlavour - Flavour of dwarf regnumbers
  ///
  namespace DWARFFlavour {
    enum {
      AMDIL_Generic = 0
    };
  }

  struct AMDILRegisterInfo : public AMDILGenRegisterInfo
  {
    TargetMachine &TM;
    const TargetInstrInfo &TII;

    AMDILRegisterInfo(TargetMachine &tm, const TargetInstrInfo &tii);
    /// Code Generation virtual methods...
    const uint16_t * getCalleeSavedRegs(const MachineFunction *MF = 0) const;

    const TargetRegisterClass* const*
      getCalleeSavedRegClasses(
          const MachineFunction *MF = 0) const;

    BitVector
      getReservedRegs(const MachineFunction &MF) const;
    BitVector
      getAllocatableSet(const MachineFunction &MF,
          const TargetRegisterClass *RC) const;

    void
      eliminateCallFramePseudoInstr(
          MachineFunction &MF,
          MachineBasicBlock &MBB,
          MachineBasicBlock::iterator I) const;
    void
      eliminateFrameIndex(MachineBasicBlock::iterator II,
          int SPAdj, RegScavenger *RS = NULL) const;

    void
      processFunctionBeforeFrameFinalized(MachineFunction &MF) const;

    // Debug information queries.
    unsigned int
      getRARegister() const;

    unsigned int
      getFrameRegister(const MachineFunction &MF) const;

    // Exception handling queries.
    unsigned int
      getEHExceptionRegister() const;
    unsigned int
      getEHHandlerRegister() const;

    int64_t
      getStackSize() const;

    virtual const TargetRegisterClass * getCFGStructurizerRegClass(MVT VT)
                                                                      const {
      return &AMDGPU::GPRI32RegClass;
    }
    private:
    mutable int64_t baseOffset;
    mutable int64_t nextFuncOffset;
  };

} // end namespace llvm

#endif // AMDILREGISTERINFO_H_
