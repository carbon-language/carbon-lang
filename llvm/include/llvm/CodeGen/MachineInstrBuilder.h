//===-- CodeGen/MachineInstBuilder.h - Simplify creation of MIs -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file exposes a function named BuildMI, which is useful for dramatically
// simplifying how MachineInstr's are created.  It allows use of code like this:
//
//   M = BuildMI(X86::ADDrr8, 2).addReg(argVal1).addReg(argVal2);
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MACHINEINSTRBUILDER_H
#define LLVM_CODEGEN_MACHINEINSTRBUILDER_H

#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {

class TargetInstrDesc;
class MDNode;

namespace RegState {
  enum {
    Define         = 0x2,
    Implicit       = 0x4,
    Kill           = 0x8,
    Dead           = 0x10,
    Undef          = 0x20,
    EarlyClobber   = 0x40,
    Debug          = 0x80,
    ImplicitDefine = Implicit | Define,
    ImplicitKill   = Implicit | Kill
  };
}

class MachineInstrBuilder {
  MachineInstr *MI;
public:
  MachineInstrBuilder() : MI(0) {}
  explicit MachineInstrBuilder(MachineInstr *mi) : MI(mi) {}

  /// Allow automatic conversion to the machine instruction we are working on.
  ///
  operator MachineInstr*() const { return MI; }
  MachineInstr *operator->() const { return MI; }
  operator MachineBasicBlock::iterator() const { return MI; }

  /// addReg - Add a new virtual register operand...
  ///
  const
  MachineInstrBuilder &addReg(unsigned RegNo, unsigned flags = 0,
                              unsigned SubReg = 0) const {
    assert((flags & 0x1) == 0 &&
           "Passing in 'true' to addReg is forbidden! Use enums instead.");
    MI->addOperand(MachineOperand::CreateReg(RegNo,
                                             flags & RegState::Define,
                                             flags & RegState::Implicit,
                                             flags & RegState::Kill,
                                             flags & RegState::Dead,
                                             flags & RegState::Undef,
                                             flags & RegState::EarlyClobber,
                                             SubReg,
                                             flags & RegState::Debug));
    return *this;
  }

  /// addImm - Add a new immediate operand.
  ///
  const MachineInstrBuilder &addImm(int64_t Val) const {
    MI->addOperand(MachineOperand::CreateImm(Val));
    return *this;
  }

  const MachineInstrBuilder &addFPImm(const ConstantFP *Val) const {
    MI->addOperand(MachineOperand::CreateFPImm(Val));
    return *this;
  }

  const MachineInstrBuilder &addMBB(MachineBasicBlock *MBB,
                                    unsigned char TargetFlags = 0) const {
    MI->addOperand(MachineOperand::CreateMBB(MBB, TargetFlags));
    return *this;
  }

  const MachineInstrBuilder &addFrameIndex(unsigned Idx) const {
    MI->addOperand(MachineOperand::CreateFI(Idx));
    return *this;
  }

  const MachineInstrBuilder &addConstantPoolIndex(unsigned Idx,
                                                  int Offset = 0,
                                          unsigned char TargetFlags = 0) const {
    MI->addOperand(MachineOperand::CreateCPI(Idx, Offset, TargetFlags));
    return *this;
  }

  const MachineInstrBuilder &addJumpTableIndex(unsigned Idx,
                                          unsigned char TargetFlags = 0) const {
    MI->addOperand(MachineOperand::CreateJTI(Idx, TargetFlags));
    return *this;
  }

  const MachineInstrBuilder &addGlobalAddress(const GlobalValue *GV,
                                              int64_t Offset = 0,
                                          unsigned char TargetFlags = 0) const {
    MI->addOperand(MachineOperand::CreateGA(GV, Offset, TargetFlags));
    return *this;
  }

  const MachineInstrBuilder &addExternalSymbol(const char *FnName,
                                          unsigned char TargetFlags = 0) const {
    MI->addOperand(MachineOperand::CreateES(FnName, TargetFlags));
    return *this;
  }

  const MachineInstrBuilder &addMemOperand(MachineMemOperand *MMO) const {
    MI->addMemOperand(*MI->getParent()->getParent(), MMO);
    return *this;
  }

  const MachineInstrBuilder &setMemRefs(MachineInstr::mmo_iterator b,
                                        MachineInstr::mmo_iterator e) const {
    MI->setMemRefs(b, e);
    return *this;
  }


  const MachineInstrBuilder &addOperand(const MachineOperand &MO) const {
    MI->addOperand(MO);
    return *this;
  }

  const MachineInstrBuilder &addMetadata(const MDNode *MD) const {
    MI->addOperand(MachineOperand::CreateMetadata(MD));
    return *this;
  }
  
  const MachineInstrBuilder &addSym(MCSymbol *Sym) const {
    MI->addOperand(MachineOperand::CreateMCSymbol(Sym));
    return *this;
  }

  const MachineInstrBuilder &setMIFlags(unsigned Flags) const {
    MI->setFlags(Flags);
    return *this;
  }

  const MachineInstrBuilder &setMIFlag(MachineInstr::MIFlag Flag) const {
    MI->setFlag(Flag);
    return *this;
  }

  // Add a displacement from an existing MachineOperand with an added offset.
  const MachineInstrBuilder &addDisp(const MachineOperand &Disp,
                                     int64_t off) const {
    switch (Disp.getType()) {
      default:
        llvm_unreachable("Unhandled operand type in addDisp()");
      case MachineOperand::MO_Immediate:
        return addImm(Disp.getImm() + off);
      case MachineOperand::MO_GlobalAddress:
        return addGlobalAddress(Disp.getGlobal(), Disp.getOffset() + off);
    }
  }
};

/// BuildMI - Builder interface.  Specify how to create the initial instruction
/// itself.
///
inline MachineInstrBuilder BuildMI(MachineFunction &MF,
                                   DebugLoc DL,
                                   const TargetInstrDesc &TID) {
  return MachineInstrBuilder(MF.CreateMachineInstr(TID, DL));
}

/// BuildMI - This version of the builder sets up the first operand as a
/// destination virtual register.
///
inline MachineInstrBuilder BuildMI(MachineFunction &MF,
                                   DebugLoc DL,
                                   const TargetInstrDesc &TID,
                                   unsigned DestReg) {
  return MachineInstrBuilder(MF.CreateMachineInstr(TID, DL))
           .addReg(DestReg, RegState::Define);
}

/// BuildMI - This version of the builder inserts the newly-built
/// instruction before the given position in the given MachineBasicBlock, and
/// sets up the first operand as a destination virtual register.
///
inline MachineInstrBuilder BuildMI(MachineBasicBlock &BB,
                                   MachineBasicBlock::iterator I,
                                   DebugLoc DL,
                                   const TargetInstrDesc &TID,
                                   unsigned DestReg) {
  MachineInstr *MI = BB.getParent()->CreateMachineInstr(TID, DL);
  BB.insert(I, MI);
  return MachineInstrBuilder(MI).addReg(DestReg, RegState::Define);
}

/// BuildMI - This version of the builder inserts the newly-built
/// instruction before the given position in the given MachineBasicBlock, and
/// does NOT take a destination register.
///
inline MachineInstrBuilder BuildMI(MachineBasicBlock &BB,
                                   MachineBasicBlock::iterator I,
                                   DebugLoc DL,
                                   const TargetInstrDesc &TID) {
  MachineInstr *MI = BB.getParent()->CreateMachineInstr(TID, DL);
  BB.insert(I, MI);
  return MachineInstrBuilder(MI);
}

/// BuildMI - This version of the builder inserts the newly-built
/// instruction at the end of the given MachineBasicBlock, and does NOT take a
/// destination register.
///
inline MachineInstrBuilder BuildMI(MachineBasicBlock *BB,
                                   DebugLoc DL,
                                   const TargetInstrDesc &TID) {
  return BuildMI(*BB, BB->end(), DL, TID);
}

/// BuildMI - This version of the builder inserts the newly-built
/// instruction at the end of the given MachineBasicBlock, and sets up the first
/// operand as a destination virtual register.
///
inline MachineInstrBuilder BuildMI(MachineBasicBlock *BB,
                                   DebugLoc DL,
                                   const TargetInstrDesc &TID,
                                   unsigned DestReg) {
  return BuildMI(*BB, BB->end(), DL, TID, DestReg);
}

inline unsigned getDefRegState(bool B) {
  return B ? RegState::Define : 0;
}
inline unsigned getImplRegState(bool B) {
  return B ? RegState::Implicit : 0;
}
inline unsigned getKillRegState(bool B) {
  return B ? RegState::Kill : 0;
}
inline unsigned getDeadRegState(bool B) {
  return B ? RegState::Dead : 0;
}
inline unsigned getUndefRegState(bool B) {
  return B ? RegState::Undef : 0;
}

} // End llvm namespace

#endif
