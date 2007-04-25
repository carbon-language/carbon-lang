//===- X86InstrInfo.cpp - X86 Instruction Information -----------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by the LLVM research group and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the X86 implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "X86InstrInfo.h"
#include "X86.h"
#include "X86GenInstrInfo.inc"
#include "X86InstrBuilder.h"
#include "X86Subtarget.h"
#include "X86TargetMachine.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/LiveVariables.h"
using namespace llvm;

X86InstrInfo::X86InstrInfo(X86TargetMachine &tm)
  : TargetInstrInfo(X86Insts, sizeof(X86Insts)/sizeof(X86Insts[0])),
    TM(tm), RI(tm, *this) {
}

bool X86InstrInfo::isMoveInstr(const MachineInstr& MI,
                               unsigned& sourceReg,
                               unsigned& destReg) const {
  MachineOpCode oc = MI.getOpcode();
  if (oc == X86::MOV8rr || oc == X86::MOV16rr ||
      oc == X86::MOV32rr || oc == X86::MOV64rr ||
      oc == X86::MOV16to16_ || oc == X86::MOV32to32_ ||
      oc == X86::FpMOV  || oc == X86::MOVSSrr || oc == X86::MOVSDrr ||
      oc == X86::FsMOVAPSrr || oc == X86::FsMOVAPDrr ||
      oc == X86::MOVAPSrr || oc == X86::MOVAPDrr ||
      oc == X86::MOVSS2PSrr || oc == X86::MOVSD2PDrr ||
      oc == X86::MOVPS2SSrr || oc == X86::MOVPD2SDrr ||
      oc == X86::MMX_MOVD64rr || oc == X86::MMX_MOVQ64rr) {
      assert(MI.getNumOperands() >= 2 &&
             MI.getOperand(0).isRegister() &&
             MI.getOperand(1).isRegister() &&
             "invalid register-register move instruction");
      sourceReg = MI.getOperand(1).getReg();
      destReg = MI.getOperand(0).getReg();
      return true;
  }
  return false;
}

unsigned X86InstrInfo::isLoadFromStackSlot(MachineInstr *MI, 
                                           int &FrameIndex) const {
  switch (MI->getOpcode()) {
  default: break;
  case X86::MOV8rm:
  case X86::MOV16rm:
  case X86::MOV16_rm:
  case X86::MOV32rm:
  case X86::MOV32_rm:
  case X86::MOV64rm:
  case X86::FpLD64m:
  case X86::MOVSSrm:
  case X86::MOVSDrm:
  case X86::MOVAPSrm:
  case X86::MOVAPDrm:
  case X86::MMX_MOVD64rm:
  case X86::MMX_MOVQ64rm:
    if (MI->getOperand(1).isFrameIndex() && MI->getOperand(2).isImmediate() &&
        MI->getOperand(3).isRegister() && MI->getOperand(4).isImmediate() &&
        MI->getOperand(2).getImmedValue() == 1 &&
        MI->getOperand(3).getReg() == 0 &&
        MI->getOperand(4).getImmedValue() == 0) {
      FrameIndex = MI->getOperand(1).getFrameIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  }
  return 0;
}

unsigned X86InstrInfo::isStoreToStackSlot(MachineInstr *MI,
                                          int &FrameIndex) const {
  switch (MI->getOpcode()) {
  default: break;
  case X86::MOV8mr:
  case X86::MOV16mr:
  case X86::MOV16_mr:
  case X86::MOV32mr:
  case X86::MOV32_mr:
  case X86::MOV64mr:
  case X86::FpSTP64m:
  case X86::MOVSSmr:
  case X86::MOVSDmr:
  case X86::MOVAPSmr:
  case X86::MOVAPDmr:
  case X86::MMX_MOVD64mr:
  case X86::MMX_MOVQ64mr:
  case X86::MMX_MOVNTQmr:
    if (MI->getOperand(0).isFrameIndex() && MI->getOperand(1).isImmediate() &&
        MI->getOperand(2).isRegister() && MI->getOperand(3).isImmediate() &&
        MI->getOperand(1).getImmedValue() == 1 &&
        MI->getOperand(2).getReg() == 0 &&
        MI->getOperand(3).getImmedValue() == 0) {
      FrameIndex = MI->getOperand(0).getFrameIndex();
      return MI->getOperand(4).getReg();
    }
    break;
  }
  return 0;
}


/// convertToThreeAddress - This method must be implemented by targets that
/// set the M_CONVERTIBLE_TO_3_ADDR flag.  When this flag is set, the target
/// may be able to convert a two-address instruction into a true
/// three-address instruction on demand.  This allows the X86 target (for
/// example) to convert ADD and SHL instructions into LEA instructions if they
/// would require register copies due to two-addressness.
///
/// This method returns a null pointer if the transformation cannot be
/// performed, otherwise it returns the new instruction.
///
MachineInstr *
X86InstrInfo::convertToThreeAddress(MachineFunction::iterator &MFI,
                                    MachineBasicBlock::iterator &MBBI,
                                    LiveVariables &LV) const {
  MachineInstr *MI = MBBI;
  // All instructions input are two-addr instructions.  Get the known operands.
  unsigned Dest = MI->getOperand(0).getReg();
  unsigned Src = MI->getOperand(1).getReg();

  MachineInstr *NewMI = NULL;
  // FIXME: 16-bit LEA's are really slow on Athlons, but not bad on P4's.  When
  // we have better subtarget support, enable the 16-bit LEA generation here.
  bool DisableLEA16 = true;

  switch (MI->getOpcode()) {
  default: return 0;
  case X86::SHUFPSrri: {
    assert(MI->getNumOperands() == 4 && "Unknown shufps instruction!");
    if (!TM.getSubtarget<X86Subtarget>().hasSSE2()) return 0;
    
    unsigned A = MI->getOperand(0).getReg();
    unsigned B = MI->getOperand(1).getReg();
    unsigned C = MI->getOperand(2).getReg();
    unsigned M = MI->getOperand(3).getImm();
    if (B != C) return 0;
    NewMI = BuildMI(get(X86::PSHUFDri), A).addReg(B).addImm(M);
    break;
  }
  case X86::SHL64ri: {
    assert(MI->getNumOperands() == 3 && "Unknown shift instruction!");
    // NOTE: LEA doesn't produce flags like shift does, but LLVM never uses
    // the flags produced by a shift yet, so this is safe.
    unsigned Dest = MI->getOperand(0).getReg();
    unsigned Src = MI->getOperand(1).getReg();
    unsigned ShAmt = MI->getOperand(2).getImm();
    if (ShAmt == 0 || ShAmt >= 4) return 0;
    
    NewMI = BuildMI(get(X86::LEA64r), Dest)
      .addReg(0).addImm(1 << ShAmt).addReg(Src).addImm(0);
    break;
  }
  case X86::SHL32ri: {
    assert(MI->getNumOperands() == 3 && "Unknown shift instruction!");
    // NOTE: LEA doesn't produce flags like shift does, but LLVM never uses
    // the flags produced by a shift yet, so this is safe.
    unsigned Dest = MI->getOperand(0).getReg();
    unsigned Src = MI->getOperand(1).getReg();
    unsigned ShAmt = MI->getOperand(2).getImm();
    if (ShAmt == 0 || ShAmt >= 4) return 0;
    
    unsigned Opc = TM.getSubtarget<X86Subtarget>().is64Bit() ?
      X86::LEA64_32r : X86::LEA32r;
    NewMI = BuildMI(get(Opc), Dest)
      .addReg(0).addImm(1 << ShAmt).addReg(Src).addImm(0);
    break;
  }
  case X86::SHL16ri: {
    assert(MI->getNumOperands() == 3 && "Unknown shift instruction!");
    if (DisableLEA16) return 0;
    
    // NOTE: LEA doesn't produce flags like shift does, but LLVM never uses
    // the flags produced by a shift yet, so this is safe.
    unsigned Dest = MI->getOperand(0).getReg();
    unsigned Src = MI->getOperand(1).getReg();
    unsigned ShAmt = MI->getOperand(2).getImm();
    if (ShAmt == 0 || ShAmt >= 4) return 0;
    
    NewMI = BuildMI(get(X86::LEA16r), Dest)
      .addReg(0).addImm(1 << ShAmt).addReg(Src).addImm(0);
    break;
  }
  }

  // FIXME: None of these instructions are promotable to LEAs without
  // additional information.  In particular, LEA doesn't set the flags that
  // add and inc do.  :(
  if (0)
  switch (MI->getOpcode()) {
  case X86::INC32r:
  case X86::INC64_32r:
    assert(MI->getNumOperands() == 2 && "Unknown inc instruction!");
    NewMI = addRegOffset(BuildMI(get(X86::LEA32r), Dest), Src, 1);
    break;
  case X86::INC16r:
  case X86::INC64_16r:
    if (DisableLEA16) return 0;
    assert(MI->getNumOperands() == 2 && "Unknown inc instruction!");
    NewMI = addRegOffset(BuildMI(get(X86::LEA16r), Dest), Src, 1);
    break;
  case X86::DEC32r:
  case X86::DEC64_32r:
    assert(MI->getNumOperands() == 2 && "Unknown dec instruction!");
    NewMI = addRegOffset(BuildMI(get(X86::LEA32r), Dest), Src, -1);
    break;
  case X86::DEC16r:
  case X86::DEC64_16r:
    if (DisableLEA16) return 0;
    assert(MI->getNumOperands() == 2 && "Unknown dec instruction!");
    NewMI = addRegOffset(BuildMI(get(X86::LEA16r), Dest), Src, -1);
    break;
  case X86::ADD32rr:
    assert(MI->getNumOperands() == 3 && "Unknown add instruction!");
    NewMI = addRegReg(BuildMI(get(X86::LEA32r), Dest), Src,
                     MI->getOperand(2).getReg());
    break;
  case X86::ADD16rr:
    if (DisableLEA16) return 0;
    assert(MI->getNumOperands() == 3 && "Unknown add instruction!");
    NewMI = addRegReg(BuildMI(get(X86::LEA16r), Dest), Src,
                     MI->getOperand(2).getReg());
    break;
  case X86::ADD32ri:
  case X86::ADD32ri8:
    assert(MI->getNumOperands() == 3 && "Unknown add instruction!");
    if (MI->getOperand(2).isImmediate())
      NewMI = addRegOffset(BuildMI(get(X86::LEA32r), Dest), Src,
                          MI->getOperand(2).getImmedValue());
    break;
  case X86::ADD16ri:
  case X86::ADD16ri8:
    if (DisableLEA16) return 0;
    assert(MI->getNumOperands() == 3 && "Unknown add instruction!");
    if (MI->getOperand(2).isImmediate())
      NewMI = addRegOffset(BuildMI(get(X86::LEA16r), Dest), Src,
                          MI->getOperand(2).getImmedValue());
    break;
  case X86::SHL16ri:
    if (DisableLEA16) return 0;
  case X86::SHL32ri:
    assert(MI->getNumOperands() == 3 && MI->getOperand(2).isImmediate() &&
           "Unknown shl instruction!");
    unsigned ShAmt = MI->getOperand(2).getImmedValue();
    if (ShAmt == 1 || ShAmt == 2 || ShAmt == 3) {
      X86AddressMode AM;
      AM.Scale = 1 << ShAmt;
      AM.IndexReg = Src;
      unsigned Opc = MI->getOpcode() == X86::SHL32ri ? X86::LEA32r :X86::LEA16r;
      NewMI = addFullAddress(BuildMI(get(Opc), Dest), AM);
    }
    break;
  }

  if (NewMI) {
    NewMI->copyKillDeadInfo(MI);
    LV.instructionChanged(MI, NewMI);  // Update live variables
    MFI->insert(MBBI, NewMI);          // Insert the new inst    
  }
  return NewMI;
}

/// commuteInstruction - We have a few instructions that must be hacked on to
/// commute them.
///
MachineInstr *X86InstrInfo::commuteInstruction(MachineInstr *MI) const {
  // FIXME: Can commute cmoves by changing the condition!
  switch (MI->getOpcode()) {
  case X86::SHRD16rri8: // A = SHRD16rri8 B, C, I -> A = SHLD16rri8 C, B, (16-I)
  case X86::SHLD16rri8: // A = SHLD16rri8 B, C, I -> A = SHRD16rri8 C, B, (16-I)
  case X86::SHRD32rri8: // A = SHRD32rri8 B, C, I -> A = SHLD32rri8 C, B, (32-I)
  case X86::SHLD32rri8:{// A = SHLD32rri8 B, C, I -> A = SHRD32rri8 C, B, (32-I)
    unsigned Opc;
    unsigned Size;
    switch (MI->getOpcode()) {
    default: assert(0 && "Unreachable!");
    case X86::SHRD16rri8: Size = 16; Opc = X86::SHLD16rri8; break;
    case X86::SHLD16rri8: Size = 16; Opc = X86::SHRD16rri8; break;
    case X86::SHRD32rri8: Size = 32; Opc = X86::SHLD32rri8; break;
    case X86::SHLD32rri8: Size = 32; Opc = X86::SHRD32rri8; break;
    }
    unsigned Amt = MI->getOperand(3).getImmedValue();
    unsigned A = MI->getOperand(0).getReg();
    unsigned B = MI->getOperand(1).getReg();
    unsigned C = MI->getOperand(2).getReg();
    bool BisKill = MI->getOperand(1).isKill();
    bool CisKill = MI->getOperand(2).isKill();
    return BuildMI(get(Opc), A).addReg(C, false, false, CisKill)
      .addReg(B, false, false, BisKill).addImm(Size-Amt);
  }
  default:
    return TargetInstrInfo::commuteInstruction(MI);
  }
}

static X86::CondCode GetCondFromBranchOpc(unsigned BrOpc) {
  switch (BrOpc) {
  default: return X86::COND_INVALID;
  case X86::JE:  return X86::COND_E;
  case X86::JNE: return X86::COND_NE;
  case X86::JL:  return X86::COND_L;
  case X86::JLE: return X86::COND_LE;
  case X86::JG:  return X86::COND_G;
  case X86::JGE: return X86::COND_GE;
  case X86::JB:  return X86::COND_B;
  case X86::JBE: return X86::COND_BE;
  case X86::JA:  return X86::COND_A;
  case X86::JAE: return X86::COND_AE;
  case X86::JS:  return X86::COND_S;
  case X86::JNS: return X86::COND_NS;
  case X86::JP:  return X86::COND_P;
  case X86::JNP: return X86::COND_NP;
  case X86::JO:  return X86::COND_O;
  case X86::JNO: return X86::COND_NO;
  }
}

unsigned X86::GetCondBranchFromCond(X86::CondCode CC) {
  switch (CC) {
  default: assert(0 && "Illegal condition code!");
  case X86::COND_E:  return X86::JE;
  case X86::COND_NE: return X86::JNE;
  case X86::COND_L:  return X86::JL;
  case X86::COND_LE: return X86::JLE;
  case X86::COND_G:  return X86::JG;
  case X86::COND_GE: return X86::JGE;
  case X86::COND_B:  return X86::JB;
  case X86::COND_BE: return X86::JBE;
  case X86::COND_A:  return X86::JA;
  case X86::COND_AE: return X86::JAE;
  case X86::COND_S:  return X86::JS;
  case X86::COND_NS: return X86::JNS;
  case X86::COND_P:  return X86::JP;
  case X86::COND_NP: return X86::JNP;
  case X86::COND_O:  return X86::JO;
  case X86::COND_NO: return X86::JNO;
  }
}

/// GetOppositeBranchCondition - Return the inverse of the specified condition,
/// e.g. turning COND_E to COND_NE.
X86::CondCode X86::GetOppositeBranchCondition(X86::CondCode CC) {
  switch (CC) {
  default: assert(0 && "Illegal condition code!");
  case X86::COND_E:  return X86::COND_NE;
  case X86::COND_NE: return X86::COND_E;
  case X86::COND_L:  return X86::COND_GE;
  case X86::COND_LE: return X86::COND_G;
  case X86::COND_G:  return X86::COND_LE;
  case X86::COND_GE: return X86::COND_L;
  case X86::COND_B:  return X86::COND_AE;
  case X86::COND_BE: return X86::COND_A;
  case X86::COND_A:  return X86::COND_BE;
  case X86::COND_AE: return X86::COND_B;
  case X86::COND_S:  return X86::COND_NS;
  case X86::COND_NS: return X86::COND_S;
  case X86::COND_P:  return X86::COND_NP;
  case X86::COND_NP: return X86::COND_P;
  case X86::COND_O:  return X86::COND_NO;
  case X86::COND_NO: return X86::COND_O;
  }
}


bool X86InstrInfo::AnalyzeBranch(MachineBasicBlock &MBB, 
                                 MachineBasicBlock *&TBB,
                                 MachineBasicBlock *&FBB,
                                 std::vector<MachineOperand> &Cond) const {
  // TODO: If FP_REG_KILL is around, ignore it.
                                   
  // If the block has no terminators, it just falls into the block after it.
  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin() || !isTerminatorInstr((--I)->getOpcode()))
    return false;

  // Get the last instruction in the block.
  MachineInstr *LastInst = I;
  
  // If there is only one terminator instruction, process it.
  if (I == MBB.begin() || !isTerminatorInstr((--I)->getOpcode())) {
    if (!isBranch(LastInst->getOpcode()))
      return true;
    
    // If the block ends with a branch there are 3 possibilities:
    // it's an unconditional, conditional, or indirect branch.
    
    if (LastInst->getOpcode() == X86::JMP) {
      TBB = LastInst->getOperand(0).getMachineBasicBlock();
      return false;
    }
    X86::CondCode BranchCode = GetCondFromBranchOpc(LastInst->getOpcode());
    if (BranchCode == X86::COND_INVALID)
      return true;  // Can't handle indirect branch.

    // Otherwise, block ends with fall-through condbranch.
    TBB = LastInst->getOperand(0).getMachineBasicBlock();
    Cond.push_back(MachineOperand::CreateImm(BranchCode));
    return false;
  }
  
  // Get the instruction before it if it's a terminator.
  MachineInstr *SecondLastInst = I;
  
  // If there are three terminators, we don't know what sort of block this is.
  if (SecondLastInst && I != MBB.begin() &&
      isTerminatorInstr((--I)->getOpcode()))
    return true;

  // If the block ends with X86::JMP and a conditional branch, handle it.
  X86::CondCode BranchCode = GetCondFromBranchOpc(SecondLastInst->getOpcode());
  if (BranchCode != X86::COND_INVALID && LastInst->getOpcode() == X86::JMP) {
    TBB = SecondLastInst->getOperand(0).getMachineBasicBlock();
    Cond.push_back(MachineOperand::CreateImm(BranchCode));
    FBB = LastInst->getOperand(0).getMachineBasicBlock();
    return false;
  }

  // Otherwise, can't handle this.
  return true;
}

void X86InstrInfo::RemoveBranch(MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin()) return;
  --I;
  if (I->getOpcode() != X86::JMP && 
      GetCondFromBranchOpc(I->getOpcode()) == X86::COND_INVALID)
    return;
  
  // Remove the branch.
  I->eraseFromParent();
  
  I = MBB.end();
  
  if (I == MBB.begin()) return;
  --I;
  if (GetCondFromBranchOpc(I->getOpcode()) == X86::COND_INVALID)
    return;
  
  // Remove the branch.
  I->eraseFromParent();
}

void X86InstrInfo::InsertBranch(MachineBasicBlock &MBB, MachineBasicBlock *TBB,
                                MachineBasicBlock *FBB,
                                const std::vector<MachineOperand> &Cond) const {
  // Shouldn't be a fall through.
  assert(TBB && "InsertBranch must not be told to insert a fallthrough");
  assert((Cond.size() == 1 || Cond.size() == 0) &&
         "X86 branch conditions have one component!");

  if (FBB == 0) { // One way branch.
    if (Cond.empty()) {
      // Unconditional branch?
      BuildMI(&MBB, get(X86::JMP)).addMBB(TBB);
    } else {
      // Conditional branch.
      unsigned Opc = GetCondBranchFromCond((X86::CondCode)Cond[0].getImm());
      BuildMI(&MBB, get(Opc)).addMBB(TBB);
    }
    return;
  }
  
  // Two-way Conditional branch.
  unsigned Opc = GetCondBranchFromCond((X86::CondCode)Cond[0].getImm());
  BuildMI(&MBB, get(Opc)).addMBB(TBB);
  BuildMI(&MBB, get(X86::JMP)).addMBB(FBB);
}

bool X86InstrInfo::BlockHasNoFallThrough(MachineBasicBlock &MBB) const {
  if (MBB.empty()) return false;
  
  switch (MBB.back().getOpcode()) {
  case X86::JMP:     // Uncond branch.
  case X86::JMP32r:  // Indirect branch.
  case X86::JMP32m:  // Indirect branch through mem.
    return true;
  default: return false;
  }
}

bool X86InstrInfo::
ReverseBranchCondition(std::vector<MachineOperand> &Cond) const {
  assert(Cond.size() == 1 && "Invalid X86 branch condition!");
  Cond[0].setImm(GetOppositeBranchCondition((X86::CondCode)Cond[0].getImm()));
  return false;
}

const TargetRegisterClass *X86InstrInfo::getPointerRegClass() const {
  const X86Subtarget *Subtarget = &TM.getSubtarget<X86Subtarget>();
  if (Subtarget->is64Bit())
    return &X86::GR64RegClass;
  else
    return &X86::GR32RegClass;
}
