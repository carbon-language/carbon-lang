//===- SPUInstrInfo.h - Cell SPU Instruction Information --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the CellSPU implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef SPU_INSTRUCTIONINFO_H
#define SPU_INSTRUCTIONINFO_H

#include "SPU.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "SPURegisterInfo.h"

namespace llvm {
  //! Cell SPU instruction information class
  class SPUInstrInfo : public TargetInstrInfo
  {
    SPUTargetMachine &TM;
    const SPURegisterInfo RI;
  public:
    SPUInstrInfo(SPUTargetMachine &tm);

    /// getRegisterInfo - TargetInstrInfo is a superset of MRegister info.  As
    /// such, whenever a client has an instance of instruction info, it should
    /// always be able to get register info as well (through this method).
    ///
    virtual const MRegisterInfo &getRegisterInfo() const { return RI; }

    /// getPointerRegClass - Return the register class to use to hold pointers.
    /// This is used for addressing modes.
    virtual const TargetRegisterClass *getPointerRegClass() const;  

    // Return true if the instruction is a register to register move and
    // leave the source and dest operands in the passed parameters.
    //
    virtual bool isMoveInstr(const MachineInstr& MI,
			     unsigned& sourceReg,
			     unsigned& destReg) const;

    unsigned isLoadFromStackSlot(MachineInstr *MI, int &FrameIndex) const;
    unsigned isStoreToStackSlot(MachineInstr *MI, int &FrameIndex) const;
  };
}

#endif
