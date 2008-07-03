//===- MipsInstrInfo.cpp - Mips Instruction Information ---------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Mips implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "Mips.h"
#include "MipsInstrInfo.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "MipsGenInstrInfo.inc"

using namespace llvm;

// TODO: Add the subtarget support on this constructor
MipsInstrInfo::MipsInstrInfo(MipsTargetMachine &tm)
  : TargetInstrInfoImpl(MipsInsts, array_lengthof(MipsInsts)),
    TM(tm), RI(*this) {}

static bool isZeroImm(const MachineOperand &op) {
  return op.isImmediate() && op.getImm() == 0;
}

/// Return true if the instruction is a register to register move and
/// leave the source and dest operands in the passed parameters.
bool MipsInstrInfo::
isMoveInstr(const MachineInstr &MI, unsigned &SrcReg, unsigned &DstReg) const 
{
  //  addu  $dst, $src, $zero || addu  $dst, $zero, $src
  //  or    $dst, $src, $zero || or    $dst, $zero, $src
  if ((MI.getOpcode() == Mips::ADDu) || (MI.getOpcode() == Mips::OR))
  {
    if (MI.getOperand(1).getReg() == Mips::ZERO) {
      DstReg = MI.getOperand(0).getReg();
      SrcReg = MI.getOperand(2).getReg();
      return true;
    } else if (MI.getOperand(2).getReg() == Mips::ZERO) {
      DstReg = MI.getOperand(0).getReg();
      SrcReg = MI.getOperand(1).getReg();
      return true;
    }
  }

  //  addiu $dst, $src, 0
  if (MI.getOpcode() == Mips::ADDiu) 
  {
    if ((MI.getOperand(1).isRegister()) && (isZeroImm(MI.getOperand(2)))) {
      DstReg = MI.getOperand(0).getReg();
      SrcReg = MI.getOperand(1).getReg();
      return true;
    }
  }
  return false;
}

/// isLoadFromStackSlot - If the specified machine instruction is a direct
/// load from a stack slot, return the virtual or physical register number of
/// the destination along with the FrameIndex of the loaded stack slot.  If
/// not, return 0.  This predicate must return 0 if the instruction has
/// any side effects other than loading from the stack slot.
unsigned MipsInstrInfo::
isLoadFromStackSlot(MachineInstr *MI, int &FrameIndex) const 
{
  if (MI->getOpcode() == Mips::LW) 
  {
    if ((MI->getOperand(2).isFrameIndex()) && // is a stack slot
        (MI->getOperand(1).isImmediate()) &&  // the imm is zero
        (isZeroImm(MI->getOperand(1)))) 
    {
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
unsigned MipsInstrInfo::
isStoreToStackSlot(MachineInstr *MI, int &FrameIndex) const 
{
  if (MI->getOpcode() == Mips::SW) {
    if ((MI->getOperand(0).isFrameIndex()) && // is a stack slot
        (MI->getOperand(1).isImmediate()) &&  // the imm is zero
        (isZeroImm(MI->getOperand(1)))) 
    {
      FrameIndex = MI->getOperand(0).getIndex();
      return MI->getOperand(2).getReg();
    }
  }
  return 0;
}

/// insertNoop - If data hazard condition is found insert the target nop
/// instruction.
void MipsInstrInfo::
insertNoop(MachineBasicBlock &MBB, MachineBasicBlock::iterator MI) const 
{
  BuildMI(MBB, MI, get(Mips::NOP));
}

//===----------------------------------------------------------------------===//
// Branch Analysis
//===----------------------------------------------------------------------===//

/// GetCondFromBranchOpc - Return the Mips CC that matches 
/// the correspondent Branch instruction opcode.
static Mips::CondCode GetCondFromBranchOpc(unsigned BrOpc) 
{
  switch (BrOpc) {
  default: return Mips::COND_INVALID;
  case Mips::BEQ  : return Mips::COND_E;
  case Mips::BNE  : return Mips::COND_NE;
  case Mips::BGTZ : return Mips::COND_GZ;
  case Mips::BGEZ : return Mips::COND_GEZ;
  case Mips::BLTZ : return Mips::COND_LZ;
  case Mips::BLEZ : return Mips::COND_LEZ;
  }
}

/// GetCondBranchFromCond - Return the Branch instruction
/// opcode that matches the cc.
unsigned Mips::GetCondBranchFromCond(Mips::CondCode CC) 
{
  switch (CC) {
  default: assert(0 && "Illegal condition code!");
  case Mips::COND_E   : return Mips::BEQ;
  case Mips::COND_NE  : return Mips::BNE;
  case Mips::COND_GZ  : return Mips::BGTZ;
  case Mips::COND_GEZ : return Mips::BGEZ;
  case Mips::COND_LZ  : return Mips::BLTZ;
  case Mips::COND_LEZ : return Mips::BLEZ;
  }
}

/// GetOppositeBranchCondition - Return the inverse of the specified 
/// condition, e.g. turning COND_E to COND_NE.
Mips::CondCode Mips::GetOppositeBranchCondition(Mips::CondCode CC) 
{
  switch (CC) {
  default: assert(0 && "Illegal condition code!");
  case Mips::COND_E   : return Mips::COND_NE;
  case Mips::COND_NE  : return Mips::COND_E;
  case Mips::COND_GZ  : return Mips::COND_LEZ;
  case Mips::COND_GEZ : return Mips::COND_LZ;
  case Mips::COND_LZ  : return Mips::COND_GEZ;
  case Mips::COND_LEZ : return Mips::COND_GZ;
  }
}

bool MipsInstrInfo::AnalyzeBranch(MachineBasicBlock &MBB, 
                                  MachineBasicBlock *&TBB,
                                  MachineBasicBlock *&FBB,
                                  std::vector<MachineOperand> &Cond) const 
{
  // If the block has no terminators, it just falls into the block after it.
  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin() || !isUnpredicatedTerminator(--I))
    return false;
  
  // Get the last instruction in the block.
  MachineInstr *LastInst = I;
  
  // If there is only one terminator instruction, process it.
  unsigned LastOpc = LastInst->getOpcode();
  if (I == MBB.begin() || !isUnpredicatedTerminator(--I)) {
    if (!LastInst->getDesc().isBranch())
      return true;

    // Unconditional branch
    if (LastOpc == Mips::J) {
      TBB = LastInst->getOperand(0).getMBB();
      return false;
    }

    Mips::CondCode BranchCode = GetCondFromBranchOpc(LastInst->getOpcode());
    if (BranchCode == Mips::COND_INVALID)
      return true;  // Can't handle indirect branch.

    // Conditional branch
    // Block ends with fall-through condbranch.
    if (LastOpc != Mips::COND_INVALID) {
      int LastNumOp = LastInst->getNumOperands();

      TBB = LastInst->getOperand(LastNumOp-1).getMBB();
      Cond.push_back(MachineOperand::CreateImm(BranchCode));

      for (int i=0; i<LastNumOp-1; i++) {
        Cond.push_back(LastInst->getOperand(i));
      }

      return false;
    }
  }
  
  // Get the instruction before it if it is a terminator.
  MachineInstr *SecondLastInst = I;
  
  // If there are three terminators, we don't know what sort of block this is.
  if (SecondLastInst && I != MBB.begin() && isUnpredicatedTerminator(--I))
    return true;

  // If the block ends with Mips::J and a Mips::BNE/Mips::BEQ, handle it.
  unsigned SecondLastOpc    = SecondLastInst->getOpcode();
  Mips::CondCode BranchCode = GetCondFromBranchOpc(SecondLastOpc);

  if (SecondLastOpc != Mips::COND_INVALID && LastOpc == Mips::J) {
    int SecondNumOp = SecondLastInst->getNumOperands();

    TBB = SecondLastInst->getOperand(SecondNumOp-1).getMBB();
    Cond.push_back(MachineOperand::CreateImm(BranchCode));

    for (int i=0; i<SecondNumOp-1; i++) {
      Cond.push_back(SecondLastInst->getOperand(i));
    }

    FBB = LastInst->getOperand(0).getMBB();
    return false;
  }
  
  // If the block ends with two unconditional branches, handle it. The last 
  // one is not executed, so remove it.
  if ((SecondLastOpc == Mips::J) && (LastOpc == Mips::J)) {
    TBB = SecondLastInst->getOperand(0).getMBB();
    I = LastInst;
    I->eraseFromParent();
    return false;
  }

  // Otherwise, can't handle this.
  return true;
}

unsigned MipsInstrInfo::
InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB, 
             MachineBasicBlock *FBB, const std::vector<MachineOperand> &Cond)
             const
{
  // Shouldn't be a fall through.
  assert(TBB && "InsertBranch must not be told to insert a fallthrough");
  assert((Cond.size() == 3 || Cond.size() == 2 || Cond.size() == 0) &&
         "Mips branch conditions can have two|three components!");

  if (FBB == 0) { // One way branch.
    if (Cond.empty()) {
      // Unconditional branch?
      BuildMI(&MBB, get(Mips::J)).addMBB(TBB);
    } else {
      // Conditional branch.
      unsigned Opc = GetCondBranchFromCond((Mips::CondCode)Cond[0].getImm());
      const TargetInstrDesc &TID = get(Opc);

      if (TID.getNumOperands() == 3)
        BuildMI(&MBB, TID).addReg(Cond[1].getReg())
                          .addReg(Cond[2].getReg())
                          .addMBB(TBB);
      else
        BuildMI(&MBB, TID).addReg(Cond[1].getReg())
                          .addMBB(TBB);

    }                             
    return 1;
  }
  
  // Two-way Conditional branch.
  unsigned Opc = GetCondBranchFromCond((Mips::CondCode)Cond[0].getImm());
  const TargetInstrDesc &TID = get(Opc);

  if (TID.getNumOperands() == 3)
    BuildMI(&MBB, TID).addReg(Cond[1].getReg()).addReg(Cond[2].getReg())
                      .addMBB(TBB);
  else
    BuildMI(&MBB, TID).addReg(Cond[1].getReg()).addMBB(TBB);

  BuildMI(&MBB, get(Mips::J)).addMBB(FBB);
  return 2;
}

void MipsInstrInfo::
copyRegToReg(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
             unsigned DestReg, unsigned SrcReg,
             const TargetRegisterClass *DestRC,
             const TargetRegisterClass *SrcRC) const {
  if (DestRC != SrcRC) {
    cerr << "Not yet supported!";
    abort();
  }

  if (DestRC == Mips::CPURegsRegisterClass)
    BuildMI(MBB, I, get(Mips::ADDu), DestReg).addReg(Mips::ZERO)
      .addReg(SrcReg);
  else
    assert (0 && "Can't copy this register");
}

void MipsInstrInfo::
storeRegToStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
          unsigned SrcReg, bool isKill, int FI, 
          const TargetRegisterClass *RC) const 
{
  if (RC == Mips::CPURegsRegisterClass)
    BuildMI(MBB, I, get(Mips::SW)).addReg(SrcReg, false, false, isKill)
          .addImm(0).addFrameIndex(FI);
  else
    assert(0 && "Can't store this register to stack slot");
}

void MipsInstrInfo::storeRegToAddr(MachineFunction &MF, unsigned SrcReg,
                                      bool isKill,
                                      SmallVectorImpl<MachineOperand> &Addr,
                                      const TargetRegisterClass *RC,
                                 SmallVectorImpl<MachineInstr*> &NewMIs) const {
  if (RC != Mips::CPURegsRegisterClass)
    assert(0 && "Can't store this register");
  MachineInstrBuilder MIB = BuildMI(get(Mips::SW))
    .addReg(SrcReg, false, false, isKill);
  for (unsigned i = 0, e = Addr.size(); i != e; ++i) {
    MachineOperand &MO = Addr[i];
    if (MO.isRegister())
      MIB.addReg(MO.getReg());
    else if (MO.isImmediate())
      MIB.addImm(MO.getImm());
    else
      MIB.addFrameIndex(MO.getIndex());
  }
  NewMIs.push_back(MIB);
  return;
}

void MipsInstrInfo::
loadRegFromStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                     unsigned DestReg, int FI,
                     const TargetRegisterClass *RC) const 
{
  if (RC == Mips::CPURegsRegisterClass)
    BuildMI(MBB, I, get(Mips::LW), DestReg).addImm(0).addFrameIndex(FI);
  else
    assert(0 && "Can't load this register from stack slot");
}

void MipsInstrInfo::loadRegFromAddr(MachineFunction &MF, unsigned DestReg,
                                       SmallVectorImpl<MachineOperand> &Addr,
                                       const TargetRegisterClass *RC,
                                 SmallVectorImpl<MachineInstr*> &NewMIs) const {
  if (RC != Mips::CPURegsRegisterClass)
    assert(0 && "Can't load this register");
  MachineInstrBuilder MIB = BuildMI(get(Mips::LW), DestReg);
  for (unsigned i = 0, e = Addr.size(); i != e; ++i) {
    MachineOperand &MO = Addr[i];
    if (MO.isRegister())
      MIB.addReg(MO.getReg());
    else if (MO.isImmediate())
      MIB.addImm(MO.getImm());
    else
      MIB.addFrameIndex(MO.getIndex());
  }
  NewMIs.push_back(MIB);
  return;
}

MachineInstr *MipsInstrInfo::
foldMemoryOperand(MachineFunction &MF,
                  MachineInstr* MI,
                  SmallVectorImpl<unsigned> &Ops, int FI) const 
{
  if (Ops.size() != 1) return NULL;

  MachineInstr *NewMI = NULL;

  switch (MI->getOpcode()) 
  {
    case Mips::ADDu:
      if ((MI->getOperand(0).isRegister()) &&
        (MI->getOperand(1).isRegister()) && 
        (MI->getOperand(1).getReg() == Mips::ZERO) &&
        (MI->getOperand(2).isRegister())) 
      {
        if (Ops[0] == 0) {    // COPY -> STORE
          unsigned SrcReg = MI->getOperand(2).getReg();
          bool isKill = MI->getOperand(2).isKill();
          NewMI = BuildMI(get(Mips::SW)).addFrameIndex(FI)
            .addImm(0).addReg(SrcReg, false, false, isKill);
        } else {              // COPY -> LOAD
          unsigned DstReg = MI->getOperand(0).getReg();
          bool isDead = MI->getOperand(0).isDead();
          NewMI = BuildMI(get(Mips::LW))
            .addReg(DstReg, true, false, false, isDead)
            .addImm(0).addFrameIndex(FI);
        }
      }
      break;
  }

  return NewMI;
}

unsigned MipsInstrInfo::
RemoveBranch(MachineBasicBlock &MBB) const 
{
  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin()) return 0;
  --I;
  if (I->getOpcode() != Mips::J && 
      GetCondFromBranchOpc(I->getOpcode()) == Mips::COND_INVALID)
    return 0;
  
  // Remove the branch.
  I->eraseFromParent();
  
  I = MBB.end();
  
  if (I == MBB.begin()) return 1;
  --I;
  if (GetCondFromBranchOpc(I->getOpcode()) == Mips::COND_INVALID)
    return 1;
  
  // Remove the branch.
  I->eraseFromParent();
  return 2;
}

/// BlockHasNoFallThrough - Analyse if MachineBasicBlock does not
/// fall-through into its successor block.
bool MipsInstrInfo::
BlockHasNoFallThrough(MachineBasicBlock &MBB) const 
{
  if (MBB.empty()) return false;
  
  switch (MBB.back().getOpcode()) {
  case Mips::RET:     // Return.
  case Mips::JR:      // Indirect branch.
  case Mips::J:       // Uncond branch.
    return true;
  default: return false;
  }
}

/// ReverseBranchCondition - Return the inverse opcode of the 
/// specified Branch instruction.
bool MipsInstrInfo::
ReverseBranchCondition(std::vector<MachineOperand> &Cond) const 
{
  assert( (Cond.size() == 3 || Cond.size() == 2) && 
          "Invalid Mips branch condition!");
  Cond[0].setImm(GetOppositeBranchCondition((Mips::CondCode)Cond[0].getImm()));
  return false;
}


