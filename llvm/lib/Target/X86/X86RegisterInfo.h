//===- X86RegisterInfo.h - X86 Register Information Impl --------*- C++ -*-===//
// 
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
// 
//===----------------------------------------------------------------------===//
//
// This file contains the X86 implementation of the MRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef X86REGISTERINFO_H
#define X86REGISTERINFO_H

#include "llvm/Target/MRegisterInfo.h"

class llvm::Type;

#include "X86GenRegisterInfo.h.inc"

namespace llvm {

struct X86RegisterInfo : public X86GenRegisterInfo {
  X86RegisterInfo();
  const TargetRegisterClass* getRegClassForType(const Type* Ty) const;

  /// Code Generation virtual methods...
  int storeRegToStackSlot(MachineBasicBlock &MBB,
                          MachineBasicBlock::iterator MI,
                          unsigned SrcReg, int FrameIndex,
                          const TargetRegisterClass *RC) const;

  int loadRegFromStackSlot(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MI,
                           unsigned DestReg, int FrameIndex,
                           const TargetRegisterClass *RC) const;
  
  int copyRegToReg(MachineBasicBlock &MBB,
                   MachineBasicBlock::iterator MI,
		   unsigned DestReg, unsigned SrcReg,
		   const TargetRegisterClass *RC) const;

  /// foldMemoryOperand - If this target supports it, fold a load or store of
  /// the specified stack slot into the specified machine instruction for the
  /// specified operand.  If this is possible, the target should perform the
  /// folding and return true, otherwise it should return false.  If it folds
  /// the instruction, it is likely that the MachineInstruction the iterator
  /// references has been changed.
  virtual MachineInstr* foldMemoryOperand(MachineInstr* MI,
                                          unsigned OpNum,
                                          int FrameIndex) const;


  void eliminateCallFramePseudoInstr(MachineFunction &MF,
                                     MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator MI) const;

  void eliminateFrameIndex(MachineFunction &MF,
                           MachineBasicBlock::iterator MI) const;

  void processFunctionBeforeFrameFinalized(MachineFunction &MF) const;

  void emitPrologue(MachineFunction &MF) const;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;
};

} // End llvm namespace

#endif
