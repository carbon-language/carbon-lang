//===- X86RegisterInfo.h - X86 Register Information Impl --------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the X86 implementation of the TargetRegisterInfo class.
//
//===----------------------------------------------------------------------===//

#ifndef X86REGISTERINFO_H
#define X86REGISTERINFO_H

#include "llvm/Target/TargetRegisterInfo.h"
#include "X86GenRegisterInfo.h.inc"

namespace llvm {
  class Type;
  class TargetInstrInfo;
  class X86TargetMachine;

/// N86 namespace - Native X86 register numbers
///
namespace N86 {
  enum {
    EAX = 0, ECX = 1, EDX = 2, EBX = 3, ESP = 4, EBP = 5, ESI = 6, EDI = 7
  };
}

/// DWARFFlavour - Flavour of dwarf regnumbers
///
namespace DWARFFlavour {
  enum {
    X86_64 = 0, X86_32_DarwinEH = 1, X86_32_Generic = 2
  };
} 
  
class X86RegisterInfo : public X86GenRegisterInfo {
public:
  X86TargetMachine &TM;
  const TargetInstrInfo &TII;

private:
  /// Is64Bit - Is the target 64-bits.
  ///
  bool Is64Bit;

  /// IsWin64 - Is the target on of win64 flavours
  ///
  bool IsWin64;

  /// SlotSize - Stack slot size in bytes.
  ///
  unsigned SlotSize;

  /// StackPtr - X86 physical register used as stack ptr.
  ///
  unsigned StackPtr;

  /// FramePtr - X86 physical register used as frame ptr.
  ///
  unsigned FramePtr;

public:
  X86RegisterInfo(X86TargetMachine &tm, const TargetInstrInfo &tii);

  /// getX86RegNum - Returns the native X86 register number for the given LLVM
  /// register identifier.
  static unsigned getX86RegNum(unsigned RegNo);

  /// getDwarfRegNum - allows modification of X86GenRegisterInfo::getDwarfRegNum
  /// (created by TableGen) for target dependencies.
  int getDwarfRegNum(unsigned RegNum, bool isEH) const;
  int getLLVMRegNum(unsigned RegNum, bool isEH) const;

  // FIXME: This should be tablegen'd like getDwarfRegNum is
  int getSEHRegNum(unsigned i) const;

  /// Code Generation virtual methods...
  /// 

  /// getMatchingSuperRegClass - Return a subclass of the specified register
  /// class A so that each register in it has a sub-register of the
  /// specified sub-register index which is in the specified register class B.
  virtual const TargetRegisterClass *
  getMatchingSuperRegClass(const TargetRegisterClass *A,
                           const TargetRegisterClass *B, unsigned Idx) const;

  const TargetRegisterClass*
  getLargestLegalSuperClass(const TargetRegisterClass *RC) const;

  /// getPointerRegClass - Returns a TargetRegisterClass used for pointer
  /// values.
  const TargetRegisterClass *getPointerRegClass(unsigned Kind = 0) const;

  /// getCrossCopyRegClass - Returns a legal register class to copy a register
  /// in the specified class to or from. Returns NULL if it is possible to copy
  /// between a two registers of the specified class.
  const TargetRegisterClass *
  getCrossCopyRegClass(const TargetRegisterClass *RC) const;

  unsigned getRegPressureLimit(const TargetRegisterClass *RC,
                               MachineFunction &MF) const;

  /// getCalleeSavedRegs - Return a null-terminated list of all of the
  /// callee-save registers on this target.
  const unsigned *getCalleeSavedRegs(const MachineFunction* MF = 0) const;

  /// getReservedRegs - Returns a bitset indexed by physical register number
  /// indicating if a register is a special register that has particular uses and
  /// should be considered unavailable at all times, e.g. SP, RA. This is used by
  /// register scavenger to determine what registers are free.
  BitVector getReservedRegs(const MachineFunction &MF) const;

  bool canRealignStack(const MachineFunction &MF) const;

  bool needsStackRealignment(const MachineFunction &MF) const;

  bool hasReservedSpillSlot(const MachineFunction &MF, unsigned Reg,
                            int &FrameIdx) const;

  void eliminateCallFramePseudoInstr(MachineFunction &MF,
                                     MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator MI) const;

  void eliminateFrameIndex(MachineBasicBlock::iterator MI,
                           int SPAdj, RegScavenger *RS = NULL) const;

  // Debug information queries.
  unsigned getRARegister() const;
  unsigned getFrameRegister(const MachineFunction &MF) const;
  unsigned getStackRegister() const { return StackPtr; }
  // FIXME: Move to FrameInfok
  unsigned getSlotSize() const { return SlotSize; }

  // Exception handling queries.
  unsigned getEHExceptionRegister() const;
  unsigned getEHHandlerRegister() const;
};

// getX86SubSuperRegister - X86 utility function. It returns the sub or super
// register of a specific X86 register.
// e.g. getX86SubSuperRegister(X86::EAX, EVT::i16) return X86:AX
unsigned getX86SubSuperRegister(unsigned, EVT, bool High=false);

} // End llvm namespace

#endif
