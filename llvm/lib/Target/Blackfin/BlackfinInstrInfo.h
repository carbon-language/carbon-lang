//===- BlackfinInstrInfo.h - Blackfin Instruction Information ---*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Blackfin implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef BLACKFININSTRUCTIONINFO_H
#define BLACKFININSTRUCTIONINFO_H

#include "llvm/Target/TargetInstrInfo.h"
#include "BlackfinRegisterInfo.h"

namespace llvm {

  class BlackfinInstrInfo : public TargetInstrInfoImpl {
    const BlackfinRegisterInfo RI;
    const BlackfinSubtarget& Subtarget;
  public:
    explicit BlackfinInstrInfo(BlackfinSubtarget &ST);

    /// getRegisterInfo - TargetInstrInfo is a superset of MRegister info.  As
    /// such, whenever a client has an instance of instruction info, it should
    /// always be able to get register info as well (through this method).
    virtual const BlackfinRegisterInfo &getRegisterInfo() const { return RI; }

    virtual unsigned isLoadFromStackSlot(const MachineInstr *MI,
                                         int &FrameIndex) const;

    virtual unsigned isStoreToStackSlot(const MachineInstr *MI,
                                        int &FrameIndex) const;

    virtual unsigned
    InsertBranch(MachineBasicBlock &MBB,
                 MachineBasicBlock *TBB,
                 MachineBasicBlock *FBB,
                 const SmallVectorImpl<MachineOperand> &Cond,
                 DebugLoc DL) const;

    virtual void copyPhysReg(MachineBasicBlock &MBB,
                             MachineBasicBlock::iterator MI, DebugLoc DL,
                             unsigned DestReg, unsigned SrcReg,
                             bool KillSrc) const;

    virtual void storeRegToStackSlot(MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator MBBI,
                                     unsigned SrcReg, bool isKill,
                                     int FrameIndex,
                                     const TargetRegisterClass *RC,
                                     const TargetRegisterInfo *TRI) const;

    virtual void storeRegToAddr(MachineFunction &MF,
                                unsigned SrcReg, bool isKill,
                                SmallVectorImpl<MachineOperand> &Addr,
                                const TargetRegisterClass *RC,
                                SmallVectorImpl<MachineInstr*> &NewMIs) const;

    virtual void loadRegFromStackSlot(MachineBasicBlock &MBB,
                                      MachineBasicBlock::iterator MBBI,
                                      unsigned DestReg, int FrameIndex,
                                      const TargetRegisterClass *RC,
                                      const TargetRegisterInfo *TRI) const;

    virtual void loadRegFromAddr(MachineFunction &MF, unsigned DestReg,
                                 SmallVectorImpl<MachineOperand> &Addr,
                                 const TargetRegisterClass *RC,
                                 SmallVectorImpl<MachineInstr*> &NewMIs) const;
  };

} // end namespace llvm

#endif
