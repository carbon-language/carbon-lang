//===- SystemZInstrBuilder.h - Functions to aid building  insts -*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file exposes functions that may be used with BuildMI from the
// MachineInstrBuilder.h file to handle SystemZ'isms in a clean way.
//
// The BuildMem function may be used with the BuildMI function to add entire
// memory references in a single, typed, function call.
//
// For reference, the order of operands for memory references is:
// (Operand), Base, Displacement, Index.
//
//===----------------------------------------------------------------------===//

#ifndef SYSTEMZINSTRBUILDER_H
#define SYSTEMZINSTRBUILDER_H

#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/PseudoSourceValue.h"

namespace llvm {

/// SystemZAddressMode - This struct holds a generalized full x86 address mode.
/// The base register can be a frame index, which will eventually be replaced
/// with R15 or R11 and Disp being offsetted accordingly.
struct SystemZAddressMode {
  enum {
    RegBase,
    FrameIndexBase
  } BaseType;

  union {
    unsigned Reg;
    int FrameIndex;
  } Base;

  unsigned IndexReg;
  int32_t Disp;
  GlobalValue *GV;

  SystemZAddressMode() : BaseType(RegBase), IndexReg(0), Disp(0) {
    Base.Reg = 0;
  }
};

/// addDirectMem - This function is used to add a direct memory reference to the
/// current instruction -- that is, a dereference of an address in a register,
/// with no index or displacement.
///
static inline const MachineInstrBuilder &
addDirectMem(const MachineInstrBuilder &MIB, unsigned Reg) {
  // Because memory references are always represented with 3
  // values, this adds: Reg, [0, NoReg] to the instruction.
  return MIB.addReg(Reg).addImm(0).addReg(0);
}

static inline const MachineInstrBuilder &
addOffset(const MachineInstrBuilder &MIB, int Offset) {
  return MIB.addImm(Offset).addReg(0);
}

/// addRegOffset - This function is used to add a memory reference of the form
/// [Reg + Offset], i.e., one with no or index, but with a
/// displacement. An example is: 10(%r15).
///
static inline const MachineInstrBuilder &
addRegOffset(const MachineInstrBuilder &MIB,
             unsigned Reg, bool isKill, int Offset) {
  return addOffset(MIB.addReg(Reg, getKillRegState(isKill)), Offset);
}

/// addRegReg - This function is used to add a memory reference of the form:
/// [Reg + Reg].
static inline const MachineInstrBuilder &
addRegReg(const MachineInstrBuilder &MIB,
            unsigned Reg1, bool isKill1, unsigned Reg2, bool isKill2) {
  return MIB.addReg(Reg1, getKillRegState(isKill1)).addImm(0)
    .addReg(Reg2, getKillRegState(isKill2));
}

static inline const MachineInstrBuilder &
addFullAddress(const MachineInstrBuilder &MIB, const SystemZAddressMode &AM) {
  if (AM.BaseType == SystemZAddressMode::RegBase)
    MIB.addReg(AM.Base.Reg);
  else if (AM.BaseType == SystemZAddressMode::FrameIndexBase)
    MIB.addFrameIndex(AM.Base.FrameIndex);
  else
    assert(0);

  return MIB.addImm(AM.Disp).addReg(AM.IndexReg);
}

/// addFrameReference - This function is used to add a reference to the base of
/// an abstract object on the stack frame of the current function.  This
/// reference has base register as the FrameIndex offset until it is resolved.
/// This allows a constant offset to be specified as well...
///
static inline const MachineInstrBuilder &
addFrameReference(const MachineInstrBuilder &MIB, int FI, int Offset = 0) {
  MachineInstr *MI = MIB;
  MachineFunction &MF = *MI->getParent()->getParent();
  MachineFrameInfo &MFI = *MF.getFrameInfo();
  const TargetInstrDesc &TID = MI->getDesc();
  unsigned Flags = 0;
  if (TID.mayLoad())
    Flags |= MachineMemOperand::MOLoad;
  if (TID.mayStore())
    Flags |= MachineMemOperand::MOStore;
  MachineMemOperand *MMO =
    MF.getMachineMemOperand(PseudoSourceValue::getFixedStack(FI),
                            Flags, Offset,
                            MFI.getObjectSize(FI),
                            MFI.getObjectAlignment(FI));
  return addOffset(MIB.addFrameIndex(FI), Offset)
            .addMemOperand(MMO);
}

} // End llvm namespace

#endif
