//===- X86RegisterInfo.h - X86 Register Information Impl ----------*-C++-*-===//
//
// This file contains the X86 implementation of the MRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef X86REGISTERINFO_H
#define X86REGISTERINFO_H

#include "llvm/Target/MRegisterInfo.h"

class Type;

struct X86RegisterInfo : public MRegisterInfo {
  X86RegisterInfo();

  MRegisterInfo::const_iterator regclass_begin() const;
  MRegisterInfo::const_iterator regclass_end() const;

  MachineBasicBlock::iterator
  storeReg2RegOffset(MachineBasicBlock &MBB,
                     MachineBasicBlock::iterator MBBI,
                     unsigned DestReg, unsigned SrcReg, 
                     unsigned ImmOffset, unsigned dataSize) const;

  MachineBasicBlock::iterator
  loadRegOffset2Reg(MachineBasicBlock &MBB,
                    MachineBasicBlock::iterator MBBI,
                    unsigned DestReg, unsigned SrcReg,
                    unsigned ImmOffset, unsigned dataSize) const;

  MachineBasicBlock::iterator
  moveReg2Reg(MachineBasicBlock &MBB,
              MachineBasicBlock::iterator MBBI,
              unsigned DestReg, unsigned SrcReg, unsigned dataSize) const;

  MachineBasicBlock::iterator
  moveImm2Reg(MachineBasicBlock &MBB,
              MachineBasicBlock::iterator MBBI,
              unsigned DestReg, unsigned Imm, unsigned dataSize) const;

  unsigned getFramePointer() const;
  unsigned getStackPointer() const;

  const unsigned* getCalleeSaveRegs() const;
  const unsigned* getCallerSaveRegs() const;

  void emitPrologue(MachineFunction &MF, unsigned numBytes) const;
  void emitEpilogue(MachineBasicBlock &MBB, unsigned numBytes) const;

  /// Returns register class appropriate for input SSA register
  /// 
  const TargetRegisterClass *getClassForReg(unsigned Reg) const;

  const TargetRegisterClass* getRegClassForType(const Type* Ty) const;

  unsigned getNumRegClasses() const;

};

#endif
