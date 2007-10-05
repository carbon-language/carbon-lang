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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Target/MRegisterInfo.h"
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

class X86RegisterInfo : public X86GenRegisterInfo {
public:
  X86TargetMachine &TM;
  const TargetInstrInfo &TII;

private:
  /// Is64Bit - Is the target 64-bits.
  ///
  bool Is64Bit;

  /// SlotSize - Stack slot size in bytes.
  ///
  unsigned SlotSize;

  /// StackPtr - X86 physical register used as stack ptr.
  ///
  unsigned StackPtr;

  /// FramePtr - X86 physical register used as frame ptr.
  ///
  unsigned FramePtr;

  /// RegOp2MemOpTable2Addr, RegOp2MemOpTable0, RegOp2MemOpTable1,
  /// RegOp2MemOpTable2 - Load / store folding opcode maps.
  ///
  DenseMap<unsigned*, unsigned> RegOp2MemOpTable2Addr;
  DenseMap<unsigned*, unsigned> RegOp2MemOpTable0;
  DenseMap<unsigned*, unsigned> RegOp2MemOpTable1;
  DenseMap<unsigned*, unsigned> RegOp2MemOpTable2;

  /// MemOp2RegOpTable - Load / store unfolding opcode map.
  ///
  DenseMap<unsigned*, std::pair<unsigned, unsigned> > MemOp2RegOpTable;

public:
  X86RegisterInfo(X86TargetMachine &tm, const TargetInstrInfo &tii);

  /// getX86RegNum - Returns the native X86 register number for the given LLVM
  /// register identifier.
  unsigned getX86RegNum(unsigned RegNo);

  /// Code Generation virtual methods...
  ///
  bool spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator MI,
                                 const std::vector<CalleeSavedInfo> &CSI) const;

  bool restoreCalleeSavedRegisters(MachineBasicBlock &MBB,
                                   MachineBasicBlock::iterator MI,
                                 const std::vector<CalleeSavedInfo> &CSI) const;

  void storeRegToStackSlot(MachineBasicBlock &MBB,
                           MachineBasicBlock::iterator MI,
                           unsigned SrcReg, int FrameIndex,
                           const TargetRegisterClass *RC) const;

  void storeRegToAddr(MachineFunction &MF, unsigned SrcReg,
                      SmallVector<MachineOperand,4> Addr,
                      const TargetRegisterClass *RC,
                      SmallVector<MachineInstr*,4> &NewMIs) const;

  void loadRegFromStackSlot(MachineBasicBlock &MBB,
                            MachineBasicBlock::iterator MI,
                            unsigned DestReg, int FrameIndex,
                            const TargetRegisterClass *RC) const;

  void loadRegFromAddr(MachineFunction &MF, unsigned DestReg,
                       SmallVector<MachineOperand,4> Addr,
                       const TargetRegisterClass *RC,
                       SmallVector<MachineInstr*,4> &NewMIs) const;

  void copyRegToReg(MachineBasicBlock &MBB,
                    MachineBasicBlock::iterator MI,
                    unsigned DestReg, unsigned SrcReg,
                    const TargetRegisterClass *DestRC,
                    const TargetRegisterClass *SrcRC) const;
 
  const TargetRegisterClass *
  getCrossCopyRegClass(const TargetRegisterClass *RC) const;

  void reMaterialize(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI,
                     unsigned DestReg, const MachineInstr *Orig) const;

  /// foldMemoryOperand - If this target supports it, fold a load or store of
  /// the specified stack slot into the specified machine instruction for the
  /// specified operand.  If this is possible, the target should perform the
  /// folding and return true, otherwise it should return false.  If it folds
  /// the instruction, it is likely that the MachineInstruction the iterator
  /// references has been changed.
  MachineInstr* foldMemoryOperand(MachineInstr* MI,
                                  unsigned OpNum,
                                  int FrameIndex) const;

  /// foldMemoryOperand - Same as the previous version except it allows folding
  /// of any load and store from / to any address, not just from a specific
  /// stack slot.
  MachineInstr* foldMemoryOperand(MachineInstr* MI,
                                  unsigned OpNum,
                                  MachineInstr* LoadMI) const;

  /// unfoldMemoryOperand - Separate a single instruction which folded a load or a
  /// a store or a load and a store into two or more instruction. If this is
  /// possible, returns true as well as the new instructions by reference.
  bool unfoldMemoryOperand(MachineFunction &MF, MachineInstr *MI,
                           SSARegMap *RegMap,
                           SmallVector<MachineInstr*, 4> &NewMIs) const;

  bool unfoldMemoryOperand(SelectionDAG &DAG, SDNode *N,
                           SmallVector<SDNode*, 4> &NewNodes) const;

  /// getCalleeSavedRegs - Return a null-terminated list of all of the
  /// callee-save registers on this target.
  const unsigned *getCalleeSavedRegs(const MachineFunction* MF = 0) const;

  /// getCalleeSavedRegClasses - Return a null-terminated list of the preferred
  /// register classes to spill each callee-saved register with.  The order and
  /// length of this list match the getCalleeSavedRegs() list.
  const TargetRegisterClass* const*
  getCalleeSavedRegClasses(const MachineFunction *MF = 0) const;

  /// getReservedRegs - Returns a bitset indexed by physical register number
  /// indicating if a register is a special register that has particular uses and
  /// should be considered unavailable at all times, e.g. SP, RA. This is used by
  /// register scavenger to determine what registers are free.
  BitVector getReservedRegs(const MachineFunction &MF) const;

  bool hasFP(const MachineFunction &MF) const;

  bool hasReservedCallFrame(MachineFunction &MF) const;

  void eliminateCallFramePseudoInstr(MachineFunction &MF,
                                     MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator MI) const;

  void eliminateFrameIndex(MachineBasicBlock::iterator MI,
                           int SPAdj, RegScavenger *RS = NULL) const;

  void processFunctionBeforeFrameFinalized(MachineFunction &MF) const;

  void emitPrologue(MachineFunction &MF) const;
  void emitEpilogue(MachineFunction &MF, MachineBasicBlock &MBB) const;

  // Debug information queries.
  unsigned getRARegister() const;
  unsigned getFrameRegister(MachineFunction &MF) const;
  void getInitialFrameState(std::vector<MachineMove> &Moves) const;

  // Exception handling queries.
  unsigned getEHExceptionRegister() const;
  unsigned getEHHandlerRegister() const;

private:
  MachineInstr* foldMemoryOperand(MachineInstr* MI,
                                  unsigned OpNum,
                                  SmallVector<MachineOperand,4> &MOs) const;
};

// getX86SubSuperRegister - X86 utility function. It returns the sub or super
// register of a specific X86 register.
// e.g. getX86SubSuperRegister(X86::EAX, MVT::i16) return X86:AX
unsigned getX86SubSuperRegister(unsigned, MVT::ValueType, bool High=false);

} // End llvm namespace

#endif
