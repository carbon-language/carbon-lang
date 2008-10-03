//===- PIC16InstrInfo.cpp - PIC16 Instruction Information -----------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source 
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the PIC16 implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "PIC16.h"
#include "PIC16InstrInfo.h"
#include "llvm/Function.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "PIC16GenInstrInfo.inc"

using namespace llvm;

// FIXME: Add the subtarget support on this constructor.
PIC16InstrInfo::PIC16InstrInfo(PIC16TargetMachine &tm)
  : TargetInstrInfoImpl(PIC16Insts, array_lengthof(PIC16Insts)),
    TM(tm), RI(*this) {}

static bool isZeroImm(const MachineOperand &op) {
  return op.isImm() && op.getImm() == 0;
}


/// isLoadFromStackSlot - If the specified machine instruction is a direct
/// load from a stack slot, return the virtual or physical register number of
/// the destination along with the FrameIndex of the loaded stack slot.  If
/// not, return 0.  This predicate must return 0 if the instruction has
/// any side effects other than loading from the stack slot.
unsigned PIC16InstrInfo::
isLoadFromStackSlot(MachineInstr *MI, int &FrameIndex) const 
{
  if (MI->getOpcode() == PIC16::MOVF) {
    if ((MI->getOperand(2).isFI()) && // is a stack slot
        (MI->getOperand(1).isImm()) &&  // the imm is zero
        (isZeroImm(MI->getOperand(1)))) {
      FrameIndex = MI->getOperand(2).getIndex();
      return MI->getOperand(0).getReg();
    }
  }

  return 0;
}

/// isStoreToStackSlot - If the specified machine instruction is a direct
/// store to a stack slot, return the virtual or physical register number of
/// the source reg along with the FrameIndex of the loaded stack slot.  If
/// not, return 0.  This predicate must return 0 if the instruction has
/// any side effects other than storing to the stack slot.
unsigned PIC16InstrInfo::
isStoreToStackSlot(MachineInstr *MI, int &FrameIndex) const 
{
  if (MI->getOpcode() == PIC16::MOVWF) {
    if ((MI->getOperand(0).isFI()) && // is a stack slot
        (MI->getOperand(1).isImm()) &&  // the imm is zero
        (isZeroImm(MI->getOperand(1)))) {
      FrameIndex = MI->getOperand(0).getIndex();
      return MI->getOperand(2).getReg();
    }
  }
  return 0;
}

void PIC16InstrInfo::
storeRegToStackSlot(MachineBasicBlock &MBB,
                    MachineBasicBlock::iterator I,
                    unsigned SrcReg, bool isKill, int FI,
                    const TargetRegisterClass *RC) const {
  const Function *Func = MBB.getParent()->getFunction();
  const std::string FuncName = Func->getName();

  char *tmpName = new char [strlen(FuncName.c_str()) +  6];
  sprintf(tmpName, "%s_tmp_%d",FuncName.c_str(),FI);

  if (RC == PIC16::CPURegsRegisterClass) {
    //src is always WREG. 
    BuildMI(MBB, I, this->get(PIC16::MOVWF))
        .addReg(SrcReg,false,false,true,true)
        .addExternalSymbol(tmpName)   // the current printer expects 3 operands,
        .addExternalSymbol(tmpName);  // all we need is actually one, 
                                      // so we repeat.
  }
  else
    assert(0 && "Can't store this register to stack slot");
}

void PIC16InstrInfo::
loadRegFromStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                     unsigned DestReg, int FI,
                     const TargetRegisterClass *RC) const 
{
  const Function *Func = MBB.getParent()->getFunction();
  const std::string FuncName = Func->getName();

  char *tmpName = new char [strlen(FuncName.c_str()) +  6];
  sprintf(tmpName, "%s_tmp_%d",FuncName.c_str(),FI);

  if (RC == PIC16::CPURegsRegisterClass)
    BuildMI(MBB, I, this->get(PIC16::MOVF), DestReg)
      .addExternalSymbol(tmpName)   // the current printer expects 3 operands,
      .addExternalSymbol(tmpName);  // all we need is actually one,so we repeat.
  else
    assert(0 && "Can't load this register from stack slot");
}

/// InsertBranch - Insert a branch into the end of the specified
/// MachineBasicBlock.  This operands to this method are the same as those
/// returned by AnalyzeBranch.  This is invoked in cases where AnalyzeBranch
/// returns success and when an unconditional branch (TBB is non-null, FBB is
/// null, Cond is empty) needs to be inserted. It returns the number of
/// instructions inserted.
unsigned PIC16InstrInfo::
InsertBranch(MachineBasicBlock &MBB, 
             MachineBasicBlock *TBB, MachineBasicBlock *FBB,
             const SmallVectorImpl<MachineOperand> &Cond) const
{
  // Shouldn't be a fall through.
  assert(TBB && "InsertBranch must not be told to insert a fallthrough");

  if (FBB == 0) { // One way branch.
    if (Cond.empty()) {
      // Unconditional branch?
      BuildMI(&MBB, get(PIC16::GOTO)).addMBB(TBB);
    } 
    return 1;
  }

  // FIXME: If the there are some conditions specified then conditional branch 
  // should be generated.
  // For the time being no instruction is being generated therefore 
  // returning NULL.
  return 0;
}

