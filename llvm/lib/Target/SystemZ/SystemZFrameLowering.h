//===-- SystemZFrameLowering.h - Frame lowering for SystemZ -----*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

#ifndef SYSTEMZFRAMELOWERING_H
#define SYSTEMZFRAMELOWERING_H

#include "SystemZSubtarget.h"
#include "llvm/ADT/IndexedMap.h"
#include "llvm/Target/TargetFrameLowering.h"

namespace llvm {
class SystemZTargetMachine;
class SystemZSubtarget;

class SystemZFrameLowering : public TargetFrameLowering {
  IndexedMap<unsigned> RegSpillOffsets;

protected:
  const SystemZTargetMachine &TM;
  const SystemZSubtarget &STI;

public:
  SystemZFrameLowering(const SystemZTargetMachine &tm,
                       const SystemZSubtarget &sti);

  // Override FrameLowering.
  virtual void
    processFunctionBeforeCalleeSavedScan(MachineFunction &MF,
                                         RegScavenger *RS) const LLVM_OVERRIDE;
  virtual bool
    spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                              MachineBasicBlock::iterator MBBI,
                              const std::vector<CalleeSavedInfo> &CSI,
                              const TargetRegisterInfo *TRI) const
    LLVM_OVERRIDE;
  virtual bool
    restoreCalleeSavedRegisters(MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator MBBII,
                                const std::vector<CalleeSavedInfo> &CSI,
                                const TargetRegisterInfo *TRI) const
    LLVM_OVERRIDE;
  virtual void emitPrologue(MachineFunction &MF) const LLVM_OVERRIDE;
  virtual void emitEpilogue(MachineFunction &MF,
                            MachineBasicBlock &MBB) const LLVM_OVERRIDE;
  virtual bool hasFP(const MachineFunction &MF) const LLVM_OVERRIDE;
  virtual int getFrameIndexOffset(const MachineFunction &MF,
                                  int FI) const LLVM_OVERRIDE;
  virtual bool hasReservedCallFrame(const MachineFunction &MF) const
    LLVM_OVERRIDE;
  virtual void
  eliminateCallFramePseudoInstr(MachineFunction &MF,
                                MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator MI) const
    LLVM_OVERRIDE;

  // The target-independent code automatically allocates save slots for
  // call-saved GPRs.  However, we don't need those slots for SystemZ,
  // because the ABI sets aside GPR save slots in the caller-allocated part
  // of the frame.  Since the target-independent code puts this unneeded
  // area at the top of the callee-allocated part of frame, we choose not
  // to allocate it and adjust the offsets accordingly.  Return the
  // size of this unallocated area.
  // FIXME: seems a bit hackish.
  uint64_t getUnallocatedTopBytes(const MachineFunction &MF) const;

  // Return the number of bytes in the callee-allocated part of the frame.
  uint64_t getAllocatedStackSize(const MachineFunction &MF) const;

  // Return the number of frame bytes that should be reserved for
  // an emergency spill slot, for use by the register scaveneger.
  // Return 0 if register scaveging won't be needed.
  unsigned getEmergencySpillSlotSize(const MachineFunction &MF) const;

  // Return the offset from the frame pointer of the emergency spill slot,
  // which always fits within a 12-bit unsigned displacement field.
  // Only valid if getEmergencySpillSlotSize(MF) returns nonzero.
  unsigned getEmergencySpillSlotOffset(const MachineFunction &MF) const;

  // Return the byte offset from the incoming stack pointer of Reg's
  // ABI-defined save slot.  Return 0 if no slot is defined for Reg.
  unsigned getRegSpillOffset(unsigned Reg) const {
    return RegSpillOffsets[Reg];
  }
};
} // end namespace llvm

#endif
