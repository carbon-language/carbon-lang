//===-- HexagonInstrInfo.cpp - Hexagon Instruction Information ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the Hexagon implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "HexagonInstrInfo.h"
#include "Hexagon.h"
#include "HexagonRegisterInfo.h"
#include "HexagonSubtarget.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/DFAPacketizer.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"
#define GET_INSTRINFO_CTOR_DTOR
#define GET_INSTRMAP_INFO
#include "HexagonGenInstrInfo.inc"
#include "HexagonGenDFAPacketizer.inc"

using namespace llvm;

///
/// Constants for Hexagon instructions.
///
const int Hexagon_MEMW_OFFSET_MAX = 4095;
const int Hexagon_MEMW_OFFSET_MIN = -4096;
const int Hexagon_MEMD_OFFSET_MAX = 8191;
const int Hexagon_MEMD_OFFSET_MIN = -8192;
const int Hexagon_MEMH_OFFSET_MAX = 2047;
const int Hexagon_MEMH_OFFSET_MIN = -2048;
const int Hexagon_MEMB_OFFSET_MAX = 1023;
const int Hexagon_MEMB_OFFSET_MIN = -1024;
const int Hexagon_ADDI_OFFSET_MAX = 32767;
const int Hexagon_ADDI_OFFSET_MIN = -32768;
const int Hexagon_MEMD_AUTOINC_MAX = 56;
const int Hexagon_MEMD_AUTOINC_MIN = -64;
const int Hexagon_MEMW_AUTOINC_MAX = 28;
const int Hexagon_MEMW_AUTOINC_MIN = -32;
const int Hexagon_MEMH_AUTOINC_MAX = 14;
const int Hexagon_MEMH_AUTOINC_MIN = -16;
const int Hexagon_MEMB_AUTOINC_MAX = 7;
const int Hexagon_MEMB_AUTOINC_MIN = -8;

// Pin the vtable to this file.
void HexagonInstrInfo::anchor() {}

HexagonInstrInfo::HexagonInstrInfo(HexagonSubtarget &ST)
  : HexagonGenInstrInfo(Hexagon::ADJCALLSTACKDOWN, Hexagon::ADJCALLSTACKUP),
    RI(ST), Subtarget(ST) {
}


/// isLoadFromStackSlot - If the specified machine instruction is a direct
/// load from a stack slot, return the virtual or physical register number of
/// the destination along with the FrameIndex of the loaded stack slot.  If
/// not, return 0.  This predicate must return 0 if the instruction has
/// any side effects other than loading from the stack slot.
unsigned HexagonInstrInfo::isLoadFromStackSlot(const MachineInstr *MI,
                                             int &FrameIndex) const {


  switch (MI->getOpcode()) {
  default: break;
  case Hexagon::LDriw:
  case Hexagon::LDrid:
  case Hexagon::LDrih:
  case Hexagon::LDrib:
  case Hexagon::LDriub:
    if (MI->getOperand(2).isFI() &&
        MI->getOperand(1).isImm() && (MI->getOperand(1).getImm() == 0)) {
      FrameIndex = MI->getOperand(2).getIndex();
      return MI->getOperand(0).getReg();
    }
    break;
  }
  return 0;
}


/// isStoreToStackSlot - If the specified machine instruction is a direct
/// store to a stack slot, return the virtual or physical register number of
/// the source reg along with the FrameIndex of the loaded stack slot.  If
/// not, return 0.  This predicate must return 0 if the instruction has
/// any side effects other than storing to the stack slot.
unsigned HexagonInstrInfo::isStoreToStackSlot(const MachineInstr *MI,
                                            int &FrameIndex) const {
  switch (MI->getOpcode()) {
  default: break;
  case Hexagon::STriw:
  case Hexagon::STrid:
  case Hexagon::STrih:
  case Hexagon::STrib:
    if (MI->getOperand(2).isFI() &&
        MI->getOperand(1).isImm() && (MI->getOperand(1).getImm() == 0)) {
      FrameIndex = MI->getOperand(0).getIndex();
      return MI->getOperand(2).getReg();
    }
    break;
  }
  return 0;
}


unsigned
HexagonInstrInfo::InsertBranch(MachineBasicBlock &MBB,MachineBasicBlock *TBB,
                             MachineBasicBlock *FBB,
                             const SmallVectorImpl<MachineOperand> &Cond,
                             DebugLoc DL) const{

    int BOpc   = Hexagon::JMP;
    int BccOpc = Hexagon::JMP_t;

    assert(TBB && "InsertBranch must not be told to insert a fallthrough");

    int regPos = 0;
    // Check if ReverseBranchCondition has asked to reverse this branch
    // If we want to reverse the branch an odd number of times, we want
    // JMP_f.
    if (!Cond.empty() && Cond[0].isImm() && Cond[0].getImm() == 0) {
      BccOpc = Hexagon::JMP_f;
      regPos = 1;
    }

    if (FBB == 0) {
      if (Cond.empty()) {
        // Due to a bug in TailMerging/CFG Optimization, we need to add a
        // special case handling of a predicated jump followed by an
        // unconditional jump. If not, Tail Merging and CFG Optimization go
        // into an infinite loop.
        MachineBasicBlock *NewTBB, *NewFBB;
        SmallVector<MachineOperand, 4> Cond;
        MachineInstr *Term = MBB.getFirstTerminator();
        if (isPredicated(Term) && !AnalyzeBranch(MBB, NewTBB, NewFBB, Cond,
                                                 false)) {
          MachineBasicBlock *NextBB =
            std::next(MachineFunction::iterator(&MBB));
          if (NewTBB == NextBB) {
            ReverseBranchCondition(Cond);
            RemoveBranch(MBB);
            return InsertBranch(MBB, TBB, 0, Cond, DL);
          }
        }
        BuildMI(&MBB, DL, get(BOpc)).addMBB(TBB);
      } else {
        BuildMI(&MBB, DL,
                get(BccOpc)).addReg(Cond[regPos].getReg()).addMBB(TBB);
      }
      return 1;
    }

    BuildMI(&MBB, DL, get(BccOpc)).addReg(Cond[regPos].getReg()).addMBB(TBB);
    BuildMI(&MBB, DL, get(BOpc)).addMBB(FBB);

    return 2;
}


bool HexagonInstrInfo::AnalyzeBranch(MachineBasicBlock &MBB,
                                     MachineBasicBlock *&TBB,
                                 MachineBasicBlock *&FBB,
                                 SmallVectorImpl<MachineOperand> &Cond,
                                 bool AllowModify) const {
  TBB = NULL;
  FBB = NULL;

  // If the block has no terminators, it just falls into the block after it.
  MachineBasicBlock::instr_iterator I = MBB.instr_end();
  if (I == MBB.instr_begin())
    return false;

  // A basic block may looks like this:
  //
  //  [   insn
  //     EH_LABEL
  //      insn
  //      insn
  //      insn
  //     EH_LABEL
  //      insn     ]
  //
  // It has two succs but does not have a terminator
  // Don't know how to handle it.
  do {
    --I;
    if (I->isEHLabel())
      return true;
  } while (I != MBB.instr_begin());

  I = MBB.instr_end();
  --I;

  while (I->isDebugValue()) {
    if (I == MBB.instr_begin())
      return false;
    --I;
  }

  // Delete the JMP if it's equivalent to a fall-through.
  if (AllowModify && I->getOpcode() == Hexagon::JMP &&
      MBB.isLayoutSuccessor(I->getOperand(0).getMBB())) {
    DEBUG(dbgs()<< "\nErasing the jump to successor block\n";);
    I->eraseFromParent();
    I = MBB.instr_end();
    if (I == MBB.instr_begin())
      return false;
    --I;
  }
  if (!isUnpredicatedTerminator(I))
    return false;

  // Get the last instruction in the block.
  MachineInstr *LastInst = I;
  MachineInstr *SecondLastInst = NULL;
  // Find one more terminator if present.
  do {
    if (&*I != LastInst && !I->isBundle() && isUnpredicatedTerminator(I)) {
      if (!SecondLastInst)
        SecondLastInst = I;
      else
        // This is a third branch.
        return true;
    }
    if (I == MBB.instr_begin())
      break;
    --I;
  } while(I);

  int LastOpcode = LastInst->getOpcode();

  bool LastOpcodeHasJMP_c = PredOpcodeHasJMP_c(LastOpcode);
  bool LastOpcodeHasNot = PredOpcodeHasNot(LastOpcode);

  // If there is only one terminator instruction, process it.
  if (LastInst && !SecondLastInst) {
    if (LastOpcode == Hexagon::JMP) {
      TBB = LastInst->getOperand(0).getMBB();
      return false;
    }
    if (LastOpcode == Hexagon::ENDLOOP0) {
      TBB = LastInst->getOperand(0).getMBB();
      Cond.push_back(LastInst->getOperand(0));
      return false;
    }
    if (LastOpcodeHasJMP_c) {
      TBB = LastInst->getOperand(1).getMBB();
      if (LastOpcodeHasNot) {
        Cond.push_back(MachineOperand::CreateImm(0));
      }
      Cond.push_back(LastInst->getOperand(0));
      return false;
    }
    // Otherwise, don't know what this is.
    return true;
  }

  int SecLastOpcode = SecondLastInst->getOpcode();

  bool SecLastOpcodeHasJMP_c = PredOpcodeHasJMP_c(SecLastOpcode);
  bool SecLastOpcodeHasNot = PredOpcodeHasNot(SecLastOpcode);
  if (SecLastOpcodeHasJMP_c && (LastOpcode == Hexagon::JMP)) {
    TBB =  SecondLastInst->getOperand(1).getMBB();
    if (SecLastOpcodeHasNot)
      Cond.push_back(MachineOperand::CreateImm(0));
    Cond.push_back(SecondLastInst->getOperand(0));
    FBB = LastInst->getOperand(0).getMBB();
    return false;
  }

  // If the block ends with two Hexagon:JMPs, handle it.  The second one is not
  // executed, so remove it.
  if (SecLastOpcode == Hexagon::JMP && LastOpcode == Hexagon::JMP) {
    TBB = SecondLastInst->getOperand(0).getMBB();
    I = LastInst;
    if (AllowModify)
      I->eraseFromParent();
    return false;
  }

  // If the block ends with an ENDLOOP, and JMP, handle it.
  if (SecLastOpcode == Hexagon::ENDLOOP0 &&
      LastOpcode == Hexagon::JMP) {
    TBB = SecondLastInst->getOperand(0).getMBB();
    Cond.push_back(SecondLastInst->getOperand(0));
    FBB = LastInst->getOperand(0).getMBB();
    return false;
  }

  // Otherwise, can't handle this.
  return true;
}


unsigned HexagonInstrInfo::RemoveBranch(MachineBasicBlock &MBB) const {
  int BOpc   = Hexagon::JMP;
  int BccOpc = Hexagon::JMP_t;
  int BccOpcNot = Hexagon::JMP_f;

  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin()) return 0;
  --I;
  if (I->getOpcode() != BOpc && I->getOpcode() != BccOpc &&
      I->getOpcode() != BccOpcNot)
    return 0;

  // Remove the branch.
  I->eraseFromParent();

  I = MBB.end();

  if (I == MBB.begin()) return 1;
  --I;
  if (I->getOpcode() != BccOpc && I->getOpcode() != BccOpcNot)
    return 1;

  // Remove the branch.
  I->eraseFromParent();
  return 2;
}


/// \brief For a comparison instruction, return the source registers in
/// \p SrcReg and \p SrcReg2 if having two register operands, and the value it
/// compares against in CmpValue. Return true if the comparison instruction
/// can be analyzed.
bool HexagonInstrInfo::analyzeCompare(const MachineInstr *MI,
                                      unsigned &SrcReg, unsigned &SrcReg2,
                                      int &Mask, int &Value) const {
  unsigned Opc = MI->getOpcode();

  // Set mask and the first source register.
  switch (Opc) {
    case Hexagon::CMPEHexagon4rr:
    case Hexagon::CMPEQri:
    case Hexagon::CMPEQrr:
    case Hexagon::CMPGT64rr:
    case Hexagon::CMPGTU64rr:
    case Hexagon::CMPGTUri:
    case Hexagon::CMPGTUrr:
    case Hexagon::CMPGTri:
    case Hexagon::CMPGTrr:
      SrcReg = MI->getOperand(1).getReg();
      Mask = ~0;
      break;
    case Hexagon::CMPbEQri_V4:
    case Hexagon::CMPbEQrr_sbsb_V4:
    case Hexagon::CMPbEQrr_ubub_V4:
    case Hexagon::CMPbGTUri_V4:
    case Hexagon::CMPbGTUrr_V4:
    case Hexagon::CMPbGTrr_V4:
      SrcReg = MI->getOperand(1).getReg();
      Mask = 0xFF;
      break;
    case Hexagon::CMPhEQri_V4:
    case Hexagon::CMPhEQrr_shl_V4:
    case Hexagon::CMPhEQrr_xor_V4:
    case Hexagon::CMPhGTUri_V4:
    case Hexagon::CMPhGTUrr_V4:
    case Hexagon::CMPhGTrr_shl_V4:
      SrcReg = MI->getOperand(1).getReg();
      Mask = 0xFFFF;
      break;
  }

  // Set the value/second source register.
  switch (Opc) {
    case Hexagon::CMPEHexagon4rr:
    case Hexagon::CMPEQrr:
    case Hexagon::CMPGT64rr:
    case Hexagon::CMPGTU64rr:
    case Hexagon::CMPGTUrr:
    case Hexagon::CMPGTrr:
    case Hexagon::CMPbEQrr_sbsb_V4:
    case Hexagon::CMPbEQrr_ubub_V4:
    case Hexagon::CMPbGTUrr_V4:
    case Hexagon::CMPbGTrr_V4:
    case Hexagon::CMPhEQrr_shl_V4:
    case Hexagon::CMPhEQrr_xor_V4:
    case Hexagon::CMPhGTUrr_V4:
    case Hexagon::CMPhGTrr_shl_V4:
      SrcReg2 = MI->getOperand(2).getReg();
      return true;

    case Hexagon::CMPEQri:
    case Hexagon::CMPGTUri:
    case Hexagon::CMPGTri:
    case Hexagon::CMPbEQri_V4:
    case Hexagon::CMPbGTUri_V4:
    case Hexagon::CMPhEQri_V4:
    case Hexagon::CMPhGTUri_V4:
      SrcReg2 = 0;
      Value = MI->getOperand(2).getImm();
      return true;
  }

  return false;
}


void HexagonInstrInfo::copyPhysReg(MachineBasicBlock &MBB,
                                 MachineBasicBlock::iterator I, DebugLoc DL,
                                 unsigned DestReg, unsigned SrcReg,
                                 bool KillSrc) const {
  if (Hexagon::IntRegsRegClass.contains(SrcReg, DestReg)) {
    BuildMI(MBB, I, DL, get(Hexagon::TFR), DestReg).addReg(SrcReg);
    return;
  }
  if (Hexagon::DoubleRegsRegClass.contains(SrcReg, DestReg)) {
    BuildMI(MBB, I, DL, get(Hexagon::TFR64), DestReg).addReg(SrcReg);
    return;
  }
  if (Hexagon::PredRegsRegClass.contains(SrcReg, DestReg)) {
    // Map Pd = Ps to Pd = or(Ps, Ps).
    BuildMI(MBB, I, DL, get(Hexagon::OR_pp),
            DestReg).addReg(SrcReg).addReg(SrcReg);
    return;
  }
  if (Hexagon::DoubleRegsRegClass.contains(DestReg) &&
      Hexagon::IntRegsRegClass.contains(SrcReg)) {
    // We can have an overlap between single and double reg: r1:0 = r0.
    if(SrcReg == RI.getSubReg(DestReg, Hexagon::subreg_loreg)) {
        // r1:0 = r0
        BuildMI(MBB, I, DL, get(Hexagon::TFRI), (RI.getSubReg(DestReg,
                Hexagon::subreg_hireg))).addImm(0);
    } else {
        // r1:0 = r1 or no overlap.
        BuildMI(MBB, I, DL, get(Hexagon::TFR), (RI.getSubReg(DestReg,
                Hexagon::subreg_loreg))).addReg(SrcReg);
        BuildMI(MBB, I, DL, get(Hexagon::TFRI), (RI.getSubReg(DestReg,
                Hexagon::subreg_hireg))).addImm(0);
    }
    return;
  }
  if (Hexagon::CRRegsRegClass.contains(DestReg) &&
      Hexagon::IntRegsRegClass.contains(SrcReg)) {
    BuildMI(MBB, I, DL, get(Hexagon::TFCR), DestReg).addReg(SrcReg);
    return;
  }
  if (Hexagon::PredRegsRegClass.contains(SrcReg) &&
      Hexagon::IntRegsRegClass.contains(DestReg)) {
    BuildMI(MBB, I, DL, get(Hexagon::TFR_RsPd), DestReg).
      addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }
  if (Hexagon::IntRegsRegClass.contains(SrcReg) &&
      Hexagon::PredRegsRegClass.contains(DestReg)) {
    BuildMI(MBB, I, DL, get(Hexagon::TFR_PdRs), DestReg).
      addReg(SrcReg, getKillRegState(KillSrc));
    return;
  }

  llvm_unreachable("Unimplemented");
}


void HexagonInstrInfo::
storeRegToStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                    unsigned SrcReg, bool isKill, int FI,
                    const TargetRegisterClass *RC,
                    const TargetRegisterInfo *TRI) const {

  DebugLoc DL = MBB.findDebugLoc(I);
  MachineFunction &MF = *MBB.getParent();
  MachineFrameInfo &MFI = *MF.getFrameInfo();
  unsigned Align = MFI.getObjectAlignment(FI);

  MachineMemOperand *MMO =
      MF.getMachineMemOperand(
                      MachinePointerInfo(PseudoSourceValue::getFixedStack(FI)),
                      MachineMemOperand::MOStore,
                      MFI.getObjectSize(FI),
                      Align);

  if (Hexagon::IntRegsRegClass.hasSubClassEq(RC)) {
    BuildMI(MBB, I, DL, get(Hexagon::STriw))
          .addFrameIndex(FI).addImm(0)
          .addReg(SrcReg, getKillRegState(isKill)).addMemOperand(MMO);
  } else if (Hexagon::DoubleRegsRegClass.hasSubClassEq(RC)) {
    BuildMI(MBB, I, DL, get(Hexagon::STrid))
          .addFrameIndex(FI).addImm(0)
          .addReg(SrcReg, getKillRegState(isKill)).addMemOperand(MMO);
  } else if (Hexagon::PredRegsRegClass.hasSubClassEq(RC)) {
    BuildMI(MBB, I, DL, get(Hexagon::STriw_pred))
          .addFrameIndex(FI).addImm(0)
          .addReg(SrcReg, getKillRegState(isKill)).addMemOperand(MMO);
  } else {
    llvm_unreachable("Unimplemented");
  }
}


void HexagonInstrInfo::storeRegToAddr(
                                 MachineFunction &MF, unsigned SrcReg,
                                 bool isKill,
                                 SmallVectorImpl<MachineOperand> &Addr,
                                 const TargetRegisterClass *RC,
                                 SmallVectorImpl<MachineInstr*> &NewMIs) const
{
  llvm_unreachable("Unimplemented");
}


void HexagonInstrInfo::
loadRegFromStackSlot(MachineBasicBlock &MBB, MachineBasicBlock::iterator I,
                     unsigned DestReg, int FI,
                     const TargetRegisterClass *RC,
                     const TargetRegisterInfo *TRI) const {
  DebugLoc DL = MBB.findDebugLoc(I);
  MachineFunction &MF = *MBB.getParent();
  MachineFrameInfo &MFI = *MF.getFrameInfo();
  unsigned Align = MFI.getObjectAlignment(FI);

  MachineMemOperand *MMO =
      MF.getMachineMemOperand(
                      MachinePointerInfo(PseudoSourceValue::getFixedStack(FI)),
                      MachineMemOperand::MOLoad,
                      MFI.getObjectSize(FI),
                      Align);
  if (RC == &Hexagon::IntRegsRegClass) {
    BuildMI(MBB, I, DL, get(Hexagon::LDriw), DestReg)
          .addFrameIndex(FI).addImm(0).addMemOperand(MMO);
  } else if (RC == &Hexagon::DoubleRegsRegClass) {
    BuildMI(MBB, I, DL, get(Hexagon::LDrid), DestReg)
          .addFrameIndex(FI).addImm(0).addMemOperand(MMO);
  } else if (RC == &Hexagon::PredRegsRegClass) {
    BuildMI(MBB, I, DL, get(Hexagon::LDriw_pred), DestReg)
          .addFrameIndex(FI).addImm(0).addMemOperand(MMO);
  } else {
    llvm_unreachable("Can't store this register to stack slot");
  }
}


void HexagonInstrInfo::loadRegFromAddr(MachineFunction &MF, unsigned DestReg,
                                        SmallVectorImpl<MachineOperand> &Addr,
                                        const TargetRegisterClass *RC,
                                 SmallVectorImpl<MachineInstr*> &NewMIs) const {
  llvm_unreachable("Unimplemented");
}


MachineInstr *HexagonInstrInfo::foldMemoryOperandImpl(MachineFunction &MF,
                                                    MachineInstr* MI,
                                          const SmallVectorImpl<unsigned> &Ops,
                                                    int FI) const {
  // Hexagon_TODO: Implement.
  return(0);
}

unsigned HexagonInstrInfo::createVR(MachineFunction* MF, MVT VT) const {

  MachineRegisterInfo &RegInfo = MF->getRegInfo();
  const TargetRegisterClass *TRC;
  if (VT == MVT::i1) {
    TRC = &Hexagon::PredRegsRegClass;
  } else if (VT == MVT::i32 || VT == MVT::f32) {
    TRC = &Hexagon::IntRegsRegClass;
  } else if (VT == MVT::i64 || VT == MVT::f64) {
    TRC = &Hexagon::DoubleRegsRegClass;
  } else {
    llvm_unreachable("Cannot handle this register class");
  }

  unsigned NewReg = RegInfo.createVirtualRegister(TRC);
  return NewReg;
}

bool HexagonInstrInfo::isExtendable(const MachineInstr *MI) const {
  // Constant extenders are allowed only for V4 and above.
  if (!Subtarget.hasV4TOps())
    return false;

  const MCInstrDesc &MID = MI->getDesc();
  const uint64_t F = MID.TSFlags;
  if ((F >> HexagonII::ExtendablePos) & HexagonII::ExtendableMask)
    return true;

  // TODO: This is largely obsolete now. Will need to be removed
  // in consecutive patches.
  switch(MI->getOpcode()) {
    // TFR_FI Remains a special case.
    case Hexagon::TFR_FI:
      return true;
    default:
      return false;
  }
  return  false;
}

// This returns true in two cases:
// - The OP code itself indicates that this is an extended instruction.
// - One of MOs has been marked with HMOTF_ConstExtended flag.
bool HexagonInstrInfo::isExtended(const MachineInstr *MI) const {
  // First check if this is permanently extended op code.
  const uint64_t F = MI->getDesc().TSFlags;
  if ((F >> HexagonII::ExtendedPos) & HexagonII::ExtendedMask)
    return true;
  // Use MO operand flags to determine if one of MI's operands
  // has HMOTF_ConstExtended flag set.
  for (MachineInstr::const_mop_iterator I = MI->operands_begin(),
       E = MI->operands_end(); I != E; ++I) {
    if (I->getTargetFlags() && HexagonII::HMOTF_ConstExtended)
      return true;
  }
  return  false;
}

bool HexagonInstrInfo::isBranch (const MachineInstr *MI) const {
  return MI->getDesc().isBranch();
}

bool HexagonInstrInfo::isNewValueInst(const MachineInstr *MI) const {
  if (isNewValueJump(MI))
    return true;

  if (isNewValueStore(MI))
    return true;

  return false;
}

bool HexagonInstrInfo::isSaveCalleeSavedRegsCall(const MachineInstr *MI) const {
  return MI->getOpcode() == Hexagon::SAVE_REGISTERS_CALL_V4;
}

bool HexagonInstrInfo::isPredicable(MachineInstr *MI) const {
  bool isPred = MI->getDesc().isPredicable();

  if (!isPred)
    return false;

  const int Opc = MI->getOpcode();

  switch(Opc) {
  case Hexagon::TFRI:
    return isInt<12>(MI->getOperand(1).getImm());

  case Hexagon::STrid:
  case Hexagon::STrid_indexed:
    return isShiftedUInt<6,3>(MI->getOperand(1).getImm());

  case Hexagon::STriw:
  case Hexagon::STriw_indexed:
  case Hexagon::STriw_nv_V4:
    return isShiftedUInt<6,2>(MI->getOperand(1).getImm());

  case Hexagon::STrih:
  case Hexagon::STrih_indexed:
  case Hexagon::STrih_nv_V4:
    return isShiftedUInt<6,1>(MI->getOperand(1).getImm());

  case Hexagon::STrib:
  case Hexagon::STrib_indexed:
  case Hexagon::STrib_nv_V4:
    return isUInt<6>(MI->getOperand(1).getImm());

  case Hexagon::LDrid:
  case Hexagon::LDrid_indexed:
    return isShiftedUInt<6,3>(MI->getOperand(2).getImm());

  case Hexagon::LDriw:
  case Hexagon::LDriw_indexed:
    return isShiftedUInt<6,2>(MI->getOperand(2).getImm());

  case Hexagon::LDrih:
  case Hexagon::LDriuh:
  case Hexagon::LDrih_indexed:
  case Hexagon::LDriuh_indexed:
    return isShiftedUInt<6,1>(MI->getOperand(2).getImm());

  case Hexagon::LDrib:
  case Hexagon::LDriub:
  case Hexagon::LDrib_indexed:
  case Hexagon::LDriub_indexed:
    return isUInt<6>(MI->getOperand(2).getImm());

  case Hexagon::POST_LDrid:
    return isShiftedInt<4,3>(MI->getOperand(3).getImm());

  case Hexagon::POST_LDriw:
    return isShiftedInt<4,2>(MI->getOperand(3).getImm());

  case Hexagon::POST_LDrih:
  case Hexagon::POST_LDriuh:
    return isShiftedInt<4,1>(MI->getOperand(3).getImm());

  case Hexagon::POST_LDrib:
  case Hexagon::POST_LDriub:
    return isInt<4>(MI->getOperand(3).getImm());

  case Hexagon::STrib_imm_V4:
  case Hexagon::STrih_imm_V4:
  case Hexagon::STriw_imm_V4:
    return (isUInt<6>(MI->getOperand(1).getImm()) &&
            isInt<6>(MI->getOperand(2).getImm()));

  case Hexagon::ADD_ri:
    return isInt<8>(MI->getOperand(2).getImm());

  case Hexagon::ASLH:
  case Hexagon::ASRH:
  case Hexagon::SXTB:
  case Hexagon::SXTH:
  case Hexagon::ZXTB:
  case Hexagon::ZXTH:
    return Subtarget.hasV4TOps();
  }

  return true;
}

// This function performs the following inversiones:
//
//  cPt    ---> cNotPt
//  cNotPt ---> cPt
//
unsigned HexagonInstrInfo::getInvertedPredicatedOpcode(const int Opc) const {
  int InvPredOpcode;
  InvPredOpcode = isPredicatedTrue(Opc) ? Hexagon::getFalsePredOpcode(Opc)
                                        : Hexagon::getTruePredOpcode(Opc);
  if (InvPredOpcode >= 0) // Valid instruction with the inverted predicate.
    return InvPredOpcode;

  switch(Opc) {
    default: llvm_unreachable("Unexpected predicated instruction");
    case Hexagon::COMBINE_rr_cPt:
      return Hexagon::COMBINE_rr_cNotPt;
    case Hexagon::COMBINE_rr_cNotPt:
      return Hexagon::COMBINE_rr_cPt;

      // Dealloc_return.
    case Hexagon::DEALLOC_RET_cPt_V4:
      return Hexagon::DEALLOC_RET_cNotPt_V4;
    case Hexagon::DEALLOC_RET_cNotPt_V4:
      return Hexagon::DEALLOC_RET_cPt_V4;
  }
}

// New Value Store instructions.
bool HexagonInstrInfo::isNewValueStore(const MachineInstr *MI) const {
  const uint64_t F = MI->getDesc().TSFlags;

  return ((F >> HexagonII::NVStorePos) & HexagonII::NVStoreMask);
}

bool HexagonInstrInfo::isNewValueStore(unsigned Opcode) const {
  const uint64_t F = get(Opcode).TSFlags;

  return ((F >> HexagonII::NVStorePos) & HexagonII::NVStoreMask);
}

int HexagonInstrInfo::
getMatchingCondBranchOpcode(int Opc, bool invertPredicate) const {
  enum Hexagon::PredSense inPredSense;
  inPredSense = invertPredicate ? Hexagon::PredSense_false :
                                  Hexagon::PredSense_true;
  int CondOpcode = Hexagon::getPredOpcode(Opc, inPredSense);
  if (CondOpcode >= 0) // Valid Conditional opcode/instruction
    return CondOpcode;

  // This switch case will be removed once all the instructions have been
  // modified to use relation maps.
  switch(Opc) {
  case Hexagon::TFRI_f:
    return !invertPredicate ? Hexagon::TFRI_cPt_f :
                              Hexagon::TFRI_cNotPt_f;
  case Hexagon::COMBINE_rr:
    return !invertPredicate ? Hexagon::COMBINE_rr_cPt :
                              Hexagon::COMBINE_rr_cNotPt;

  // Word.
  case Hexagon::STriw_f:
    return !invertPredicate ? Hexagon::STriw_cPt :
                              Hexagon::STriw_cNotPt;
  case Hexagon::STriw_indexed_f:
    return !invertPredicate ? Hexagon::STriw_indexed_cPt :
                              Hexagon::STriw_indexed_cNotPt;

  // DEALLOC_RETURN.
  case Hexagon::DEALLOC_RET_V4:
    return !invertPredicate ? Hexagon::DEALLOC_RET_cPt_V4 :
                              Hexagon::DEALLOC_RET_cNotPt_V4;
  }
  llvm_unreachable("Unexpected predicable instruction");
}


bool HexagonInstrInfo::
PredicateInstruction(MachineInstr *MI,
                     const SmallVectorImpl<MachineOperand> &Cond) const {
  int Opc = MI->getOpcode();
  assert (isPredicable(MI) && "Expected predicable instruction");
  bool invertJump = (!Cond.empty() && Cond[0].isImm() &&
                     (Cond[0].getImm() == 0));

  // This will change MI's opcode to its predicate version.
  // However, its operand list is still the old one, i.e. the
  // non-predicate one.
  MI->setDesc(get(getMatchingCondBranchOpcode(Opc, invertJump)));

  int oper = -1;
  unsigned int GAIdx = 0;

  // Indicates whether the current MI has a GlobalAddress operand
  bool hasGAOpnd = false;
  std::vector<MachineOperand> tmpOpnds;

  // Indicates whether we need to shift operands to right.
  bool needShift = true;

  // The predicate is ALWAYS the FIRST input operand !!!
  if (MI->getNumOperands() == 0) {
    // The non-predicate version of MI does not take any operands,
    // i.e. no outs and no ins. In this condition, the predicate
    // operand will be directly placed at Operands[0]. No operand
    // shift is needed.
    // Example: BARRIER
    needShift = false;
    oper = -1;
  }
  else if (   MI->getOperand(MI->getNumOperands()-1).isReg()
           && MI->getOperand(MI->getNumOperands()-1).isDef()
           && !MI->getOperand(MI->getNumOperands()-1).isImplicit()) {
    // The non-predicate version of MI does not have any input operands.
    // In this condition, we extend the length of Operands[] by one and
    // copy the original last operand to the newly allocated slot.
    // At this moment, it is just a place holder. Later, we will put
    // predicate operand directly into it. No operand shift is needed.
    // Example: r0=BARRIER (this is a faked insn used here for illustration)
    MI->addOperand(MI->getOperand(MI->getNumOperands()-1));
    needShift = false;
    oper = MI->getNumOperands() - 2;
  }
  else {
    // We need to right shift all input operands by one. Duplicate the
    // last operand into the newly allocated slot.
    MI->addOperand(MI->getOperand(MI->getNumOperands()-1));
  }

  if (needShift)
  {
    // Operands[ MI->getNumOperands() - 2 ] has been copied into
    // Operands[ MI->getNumOperands() - 1 ], so we start from
    // Operands[ MI->getNumOperands() - 3 ].
    // oper is a signed int.
    // It is ok if "MI->getNumOperands()-3" is -3, -2, or -1.
    for (oper = MI->getNumOperands() - 3; oper >= 0; --oper)
    {
      MachineOperand &MO = MI->getOperand(oper);

      // Opnd[0] Opnd[1] Opnd[2] Opnd[3] Opnd[4]   Opnd[5]   Opnd[6]   Opnd[7]
      // <Def0>  <Def1>  <Use0>  <Use1>  <ImpDef0> <ImpDef1> <ImpUse0> <ImpUse1>
      //               /\~
      //              /||\~
      //               ||
      //        Predicate Operand here
      if (MO.isReg() && !MO.isUse() && !MO.isImplicit()) {
        break;
      }
      if (MO.isReg()) {
        MI->getOperand(oper+1).ChangeToRegister(MO.getReg(), MO.isDef(),
                                                MO.isImplicit(), MO.isKill(),
                                                MO.isDead(), MO.isUndef(),
                                                MO.isDebug());
      }
      else if (MO.isImm()) {
        MI->getOperand(oper+1).ChangeToImmediate(MO.getImm());
      }
      else if (MO.isGlobal()) {
        // MI can not have more than one GlobalAddress operand.
        assert(hasGAOpnd == false && "MI can only have one GlobalAddress opnd");

        // There is no member function called "ChangeToGlobalAddress" in the
        // MachineOperand class (not like "ChangeToRegister" and
        // "ChangeToImmediate"). So we have to remove them from Operands[] list
        // first, and then add them back after we have inserted the predicate
        // operand. tmpOpnds[] is to remember these operands before we remove
        // them.
        tmpOpnds.push_back(MO);

        // Operands[oper] is a GlobalAddress operand;
        // Operands[oper+1] has been copied into Operands[oper+2];
        hasGAOpnd = true;
        GAIdx = oper;
        continue;
      }
      else {
        assert(false && "Unexpected operand type");
      }
    }
  }

  int regPos = invertJump ? 1 : 0;
  MachineOperand PredMO = Cond[regPos];

  // [oper] now points to the last explicit Def. Predicate operand must be
  // located at [oper+1]. See diagram above.
  // This assumes that the predicate is always the first operand,
  // i.e. Operands[0+numResults], in the set of inputs
  // It is better to have an assert here to check this. But I don't know how
  // to write this assert because findFirstPredOperandIdx() would return -1
  if (oper < -1) oper = -1;

  MI->getOperand(oper+1).ChangeToRegister(PredMO.getReg(), PredMO.isDef(),
                                          PredMO.isImplicit(), false,
                                          PredMO.isDead(), PredMO.isUndef(),
                                          PredMO.isDebug());

  MachineRegisterInfo &RegInfo = MI->getParent()->getParent()->getRegInfo();
  RegInfo.clearKillFlags(PredMO.getReg());

  if (hasGAOpnd)
  {
    unsigned int i;

    // Operands[GAIdx] is the original GlobalAddress operand, which is
    // already copied into tmpOpnds[0].
    // Operands[GAIdx] now stores a copy of Operands[GAIdx-1]
    // Operands[GAIdx+1] has already been copied into Operands[GAIdx+2],
    // so we start from [GAIdx+2]
    for (i = GAIdx + 2; i < MI->getNumOperands(); ++i)
      tmpOpnds.push_back(MI->getOperand(i));

    // Remove all operands in range [ (GAIdx+1) ... (MI->getNumOperands()-1) ]
    // It is very important that we always remove from the end of Operands[]
    // MI->getNumOperands() is at least 2 if program goes to here.
    for (i = MI->getNumOperands() - 1; i > GAIdx; --i)
      MI->RemoveOperand(i);

    for (i = 0; i < tmpOpnds.size(); ++i)
      MI->addOperand(tmpOpnds[i]);
  }

  return true;
}


bool
HexagonInstrInfo::
isProfitableToIfCvt(MachineBasicBlock &MBB,
                    unsigned NumCycles,
                    unsigned ExtraPredCycles,
                    const BranchProbability &Probability) const {
  return true;
}


bool
HexagonInstrInfo::
isProfitableToIfCvt(MachineBasicBlock &TMBB,
                    unsigned NumTCycles,
                    unsigned ExtraTCycles,
                    MachineBasicBlock &FMBB,
                    unsigned NumFCycles,
                    unsigned ExtraFCycles,
                    const BranchProbability &Probability) const {
  return true;
}

// Returns true if an instruction is predicated irrespective of the predicate
// sense. For example, all of the following will return true.
// if (p0) R1 = add(R2, R3)
// if (!p0) R1 = add(R2, R3)
// if (p0.new) R1 = add(R2, R3)
// if (!p0.new) R1 = add(R2, R3)
bool HexagonInstrInfo::isPredicated(const MachineInstr *MI) const {
  const uint64_t F = MI->getDesc().TSFlags;

  return ((F >> HexagonII::PredicatedPos) & HexagonII::PredicatedMask);
}

bool HexagonInstrInfo::isPredicated(unsigned Opcode) const {
  const uint64_t F = get(Opcode).TSFlags;

  return ((F >> HexagonII::PredicatedPos) & HexagonII::PredicatedMask);
}

bool HexagonInstrInfo::isPredicatedTrue(const MachineInstr *MI) const {
  const uint64_t F = MI->getDesc().TSFlags;

  assert(isPredicated(MI));
  return (!((F >> HexagonII::PredicatedFalsePos) &
            HexagonII::PredicatedFalseMask));
}

bool HexagonInstrInfo::isPredicatedTrue(unsigned Opcode) const {
  const uint64_t F = get(Opcode).TSFlags;

  // Make sure that the instruction is predicated.
  assert((F>> HexagonII::PredicatedPos) & HexagonII::PredicatedMask);
  return (!((F >> HexagonII::PredicatedFalsePos) &
            HexagonII::PredicatedFalseMask));
}

bool HexagonInstrInfo::isPredicatedNew(const MachineInstr *MI) const {
  const uint64_t F = MI->getDesc().TSFlags;

  assert(isPredicated(MI));
  return ((F >> HexagonII::PredicatedNewPos) & HexagonII::PredicatedNewMask);
}

bool HexagonInstrInfo::isPredicatedNew(unsigned Opcode) const {
  const uint64_t F = get(Opcode).TSFlags;

  assert(isPredicated(Opcode));
  return ((F >> HexagonII::PredicatedNewPos) & HexagonII::PredicatedNewMask);
}

// Returns true, if a ST insn can be promoted to a new-value store.
bool HexagonInstrInfo::mayBeNewStore(const MachineInstr *MI) const {
  const HexagonRegisterInfo& QRI = getRegisterInfo();
  const uint64_t F = MI->getDesc().TSFlags;

  return ((F >> HexagonII::mayNVStorePos) &
           HexagonII::mayNVStoreMask &
           QRI.Subtarget.hasV4TOps());
}

bool
HexagonInstrInfo::DefinesPredicate(MachineInstr *MI,
                                   std::vector<MachineOperand> &Pred) const {
  for (unsigned oper = 0; oper < MI->getNumOperands(); ++oper) {
    MachineOperand MO = MI->getOperand(oper);
    if (MO.isReg() && MO.isDef()) {
      const TargetRegisterClass* RC = RI.getMinimalPhysRegClass(MO.getReg());
      if (RC == &Hexagon::PredRegsRegClass) {
        Pred.push_back(MO);
        return true;
      }
    }
  }
  return false;
}


bool
HexagonInstrInfo::
SubsumesPredicate(const SmallVectorImpl<MachineOperand> &Pred1,
                  const SmallVectorImpl<MachineOperand> &Pred2) const {
  // TODO: Fix this
  return false;
}


//
// We indicate that we want to reverse the branch by
// inserting a 0 at the beginning of the Cond vector.
//
bool HexagonInstrInfo::
ReverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const {
  if (!Cond.empty() && Cond[0].isImm() && Cond[0].getImm() == 0) {
    Cond.erase(Cond.begin());
  } else {
    Cond.insert(Cond.begin(), MachineOperand::CreateImm(0));
  }
  return false;
}


bool HexagonInstrInfo::
isProfitableToDupForIfCvt(MachineBasicBlock &MBB,unsigned NumInstrs,
                          const BranchProbability &Probability) const {
  return (NumInstrs <= 4);
}

bool HexagonInstrInfo::isDeallocRet(const MachineInstr *MI) const {
  switch (MI->getOpcode()) {
  default: return false;
  case Hexagon::DEALLOC_RET_V4 :
  case Hexagon::DEALLOC_RET_cPt_V4 :
  case Hexagon::DEALLOC_RET_cNotPt_V4 :
  case Hexagon::DEALLOC_RET_cdnPnt_V4 :
  case Hexagon::DEALLOC_RET_cNotdnPnt_V4 :
  case Hexagon::DEALLOC_RET_cdnPt_V4 :
  case Hexagon::DEALLOC_RET_cNotdnPt_V4 :
   return true;
  }
}


bool HexagonInstrInfo::
isValidOffset(const int Opcode, const int Offset) const {
  // This function is to check whether the "Offset" is in the correct range of
  // the given "Opcode". If "Offset" is not in the correct range, "ADD_ri" is
  // inserted to calculate the final address. Due to this reason, the function
  // assumes that the "Offset" has correct alignment.
  // We used to assert if the offset was not properly aligned, however,
  // there are cases where a misaligned pointer recast can cause this
  // problem, and we need to allow for it. The front end warns of such
  // misaligns with respect to load size.

  switch(Opcode) {

  case Hexagon::LDriw:
  case Hexagon::LDriw_indexed:
  case Hexagon::LDriw_f:
  case Hexagon::STriw_indexed:
  case Hexagon::STriw:
  case Hexagon::STriw_f:
    return (Offset >= Hexagon_MEMW_OFFSET_MIN) &&
      (Offset <= Hexagon_MEMW_OFFSET_MAX);

  case Hexagon::LDrid:
  case Hexagon::LDrid_indexed:
  case Hexagon::LDrid_f:
  case Hexagon::STrid:
  case Hexagon::STrid_indexed:
  case Hexagon::STrid_f:
    return (Offset >= Hexagon_MEMD_OFFSET_MIN) &&
      (Offset <= Hexagon_MEMD_OFFSET_MAX);

  case Hexagon::LDrih:
  case Hexagon::LDriuh:
  case Hexagon::STrih:
    return (Offset >= Hexagon_MEMH_OFFSET_MIN) &&
      (Offset <= Hexagon_MEMH_OFFSET_MAX);

  case Hexagon::LDrib:
  case Hexagon::STrib:
  case Hexagon::LDriub:
    return (Offset >= Hexagon_MEMB_OFFSET_MIN) &&
      (Offset <= Hexagon_MEMB_OFFSET_MAX);

  case Hexagon::ADD_ri:
  case Hexagon::TFR_FI:
    return (Offset >= Hexagon_ADDI_OFFSET_MIN) &&
      (Offset <= Hexagon_ADDI_OFFSET_MAX);

  case Hexagon::MemOPw_ADDi_V4 :
  case Hexagon::MemOPw_SUBi_V4 :
  case Hexagon::MemOPw_ADDr_V4 :
  case Hexagon::MemOPw_SUBr_V4 :
  case Hexagon::MemOPw_ANDr_V4 :
  case Hexagon::MemOPw_ORr_V4 :
    return (0 <= Offset && Offset <= 255);

  case Hexagon::MemOPh_ADDi_V4 :
  case Hexagon::MemOPh_SUBi_V4 :
  case Hexagon::MemOPh_ADDr_V4 :
  case Hexagon::MemOPh_SUBr_V4 :
  case Hexagon::MemOPh_ANDr_V4 :
  case Hexagon::MemOPh_ORr_V4 :
    return (0 <= Offset && Offset <= 127);

  case Hexagon::MemOPb_ADDi_V4 :
  case Hexagon::MemOPb_SUBi_V4 :
  case Hexagon::MemOPb_ADDr_V4 :
  case Hexagon::MemOPb_SUBr_V4 :
  case Hexagon::MemOPb_ANDr_V4 :
  case Hexagon::MemOPb_ORr_V4 :
    return (0 <= Offset && Offset <= 63);

  // LDri_pred and STriw_pred are pseudo operations, so it has to take offset of
  // any size. Later pass knows how to handle it.
  case Hexagon::STriw_pred:
  case Hexagon::LDriw_pred:
    return true;

  case Hexagon::LOOP0_i:
    return isUInt<10>(Offset);

  // INLINEASM is very special.
  case Hexagon::INLINEASM:
    return true;
  }

  llvm_unreachable("No offset range is defined for this opcode. "
                   "Please define it in the above switch statement!");
}


//
// Check if the Offset is a valid auto-inc imm by Load/Store Type.
//
bool HexagonInstrInfo::
isValidAutoIncImm(const EVT VT, const int Offset) const {

  if (VT == MVT::i64) {
      return (Offset >= Hexagon_MEMD_AUTOINC_MIN &&
              Offset <= Hexagon_MEMD_AUTOINC_MAX &&
              (Offset & 0x7) == 0);
  }
  if (VT == MVT::i32) {
      return (Offset >= Hexagon_MEMW_AUTOINC_MIN &&
              Offset <= Hexagon_MEMW_AUTOINC_MAX &&
              (Offset & 0x3) == 0);
  }
  if (VT == MVT::i16) {
      return (Offset >= Hexagon_MEMH_AUTOINC_MIN &&
              Offset <= Hexagon_MEMH_AUTOINC_MAX &&
              (Offset & 0x1) == 0);
  }
  if (VT == MVT::i8) {
      return (Offset >= Hexagon_MEMB_AUTOINC_MIN &&
              Offset <= Hexagon_MEMB_AUTOINC_MAX);
  }
  llvm_unreachable("Not an auto-inc opc!");
}


bool HexagonInstrInfo::
isMemOp(const MachineInstr *MI) const {
//  return MI->getDesc().mayLoad() && MI->getDesc().mayStore();

  switch (MI->getOpcode())
  {
    default: return false;
    case Hexagon::MemOPw_ADDi_V4 :
    case Hexagon::MemOPw_SUBi_V4 :
    case Hexagon::MemOPw_ADDr_V4 :
    case Hexagon::MemOPw_SUBr_V4 :
    case Hexagon::MemOPw_ANDr_V4 :
    case Hexagon::MemOPw_ORr_V4 :
    case Hexagon::MemOPh_ADDi_V4 :
    case Hexagon::MemOPh_SUBi_V4 :
    case Hexagon::MemOPh_ADDr_V4 :
    case Hexagon::MemOPh_SUBr_V4 :
    case Hexagon::MemOPh_ANDr_V4 :
    case Hexagon::MemOPh_ORr_V4 :
    case Hexagon::MemOPb_ADDi_V4 :
    case Hexagon::MemOPb_SUBi_V4 :
    case Hexagon::MemOPb_ADDr_V4 :
    case Hexagon::MemOPb_SUBr_V4 :
    case Hexagon::MemOPb_ANDr_V4 :
    case Hexagon::MemOPb_ORr_V4 :
    case Hexagon::MemOPb_SETBITi_V4:
    case Hexagon::MemOPh_SETBITi_V4:
    case Hexagon::MemOPw_SETBITi_V4:
    case Hexagon::MemOPb_CLRBITi_V4:
    case Hexagon::MemOPh_CLRBITi_V4:
    case Hexagon::MemOPw_CLRBITi_V4:
    return true;
  }
  return false;
}


bool HexagonInstrInfo::
isSpillPredRegOp(const MachineInstr *MI) const {
  switch (MI->getOpcode()) {
    default: return false;
    case Hexagon::STriw_pred :
    case Hexagon::LDriw_pred :
      return true;
  }
}

bool HexagonInstrInfo::isNewValueJumpCandidate(const MachineInstr *MI) const {
  switch (MI->getOpcode()) {
    default: return false;
    case Hexagon::CMPEQrr:
    case Hexagon::CMPEQri:
    case Hexagon::CMPGTrr:
    case Hexagon::CMPGTri:
    case Hexagon::CMPGTUrr:
    case Hexagon::CMPGTUri:
      return true;
  }
}

bool HexagonInstrInfo::
isConditionalTransfer (const MachineInstr *MI) const {
  switch (MI->getOpcode()) {
    default: return false;
    case Hexagon::TFR_cPt:
    case Hexagon::TFR_cNotPt:
    case Hexagon::TFRI_cPt:
    case Hexagon::TFRI_cNotPt:
    case Hexagon::TFR_cdnPt:
    case Hexagon::TFR_cdnNotPt:
    case Hexagon::TFRI_cdnPt:
    case Hexagon::TFRI_cdnNotPt:
      return true;
  }
}

bool HexagonInstrInfo::isConditionalALU32 (const MachineInstr* MI) const {
  const HexagonRegisterInfo& QRI = getRegisterInfo();
  switch (MI->getOpcode())
  {
    default: return false;
    case Hexagon::ADD_ri_cPt:
    case Hexagon::ADD_ri_cNotPt:
    case Hexagon::ADD_rr_cPt:
    case Hexagon::ADD_rr_cNotPt:
    case Hexagon::XOR_rr_cPt:
    case Hexagon::XOR_rr_cNotPt:
    case Hexagon::AND_rr_cPt:
    case Hexagon::AND_rr_cNotPt:
    case Hexagon::OR_rr_cPt:
    case Hexagon::OR_rr_cNotPt:
    case Hexagon::SUB_rr_cPt:
    case Hexagon::SUB_rr_cNotPt:
    case Hexagon::COMBINE_rr_cPt:
    case Hexagon::COMBINE_rr_cNotPt:
      return true;
    case Hexagon::ASLH_cPt_V4:
    case Hexagon::ASLH_cNotPt_V4:
    case Hexagon::ASRH_cPt_V4:
    case Hexagon::ASRH_cNotPt_V4:
    case Hexagon::SXTB_cPt_V4:
    case Hexagon::SXTB_cNotPt_V4:
    case Hexagon::SXTH_cPt_V4:
    case Hexagon::SXTH_cNotPt_V4:
    case Hexagon::ZXTB_cPt_V4:
    case Hexagon::ZXTB_cNotPt_V4:
    case Hexagon::ZXTH_cPt_V4:
    case Hexagon::ZXTH_cNotPt_V4:
      return QRI.Subtarget.hasV4TOps();
  }
}

bool HexagonInstrInfo::
isConditionalLoad (const MachineInstr* MI) const {
  const HexagonRegisterInfo& QRI = getRegisterInfo();
  switch (MI->getOpcode())
  {
    default: return false;
    case Hexagon::LDrid_cPt :
    case Hexagon::LDrid_cNotPt :
    case Hexagon::LDrid_indexed_cPt :
    case Hexagon::LDrid_indexed_cNotPt :
    case Hexagon::LDriw_cPt :
    case Hexagon::LDriw_cNotPt :
    case Hexagon::LDriw_indexed_cPt :
    case Hexagon::LDriw_indexed_cNotPt :
    case Hexagon::LDrih_cPt :
    case Hexagon::LDrih_cNotPt :
    case Hexagon::LDrih_indexed_cPt :
    case Hexagon::LDrih_indexed_cNotPt :
    case Hexagon::LDrib_cPt :
    case Hexagon::LDrib_cNotPt :
    case Hexagon::LDrib_indexed_cPt :
    case Hexagon::LDrib_indexed_cNotPt :
    case Hexagon::LDriuh_cPt :
    case Hexagon::LDriuh_cNotPt :
    case Hexagon::LDriuh_indexed_cPt :
    case Hexagon::LDriuh_indexed_cNotPt :
    case Hexagon::LDriub_cPt :
    case Hexagon::LDriub_cNotPt :
    case Hexagon::LDriub_indexed_cPt :
    case Hexagon::LDriub_indexed_cNotPt :
      return true;
    case Hexagon::POST_LDrid_cPt :
    case Hexagon::POST_LDrid_cNotPt :
    case Hexagon::POST_LDriw_cPt :
    case Hexagon::POST_LDriw_cNotPt :
    case Hexagon::POST_LDrih_cPt :
    case Hexagon::POST_LDrih_cNotPt :
    case Hexagon::POST_LDrib_cPt :
    case Hexagon::POST_LDrib_cNotPt :
    case Hexagon::POST_LDriuh_cPt :
    case Hexagon::POST_LDriuh_cNotPt :
    case Hexagon::POST_LDriub_cPt :
    case Hexagon::POST_LDriub_cNotPt :
      return QRI.Subtarget.hasV4TOps();
    case Hexagon::LDrid_indexed_shl_cPt_V4 :
    case Hexagon::LDrid_indexed_shl_cNotPt_V4 :
    case Hexagon::LDrib_indexed_shl_cPt_V4 :
    case Hexagon::LDrib_indexed_shl_cNotPt_V4 :
    case Hexagon::LDriub_indexed_shl_cPt_V4 :
    case Hexagon::LDriub_indexed_shl_cNotPt_V4 :
    case Hexagon::LDrih_indexed_shl_cPt_V4 :
    case Hexagon::LDrih_indexed_shl_cNotPt_V4 :
    case Hexagon::LDriuh_indexed_shl_cPt_V4 :
    case Hexagon::LDriuh_indexed_shl_cNotPt_V4 :
    case Hexagon::LDriw_indexed_shl_cPt_V4 :
    case Hexagon::LDriw_indexed_shl_cNotPt_V4 :
      return QRI.Subtarget.hasV4TOps();
  }
}

// Returns true if an instruction is a conditional store.
//
// Note: It doesn't include conditional new-value stores as they can't be
// converted to .new predicate.
//
//               p.new NV store [ if(p0.new)memw(R0+#0)=R2.new ]
//                ^           ^
//               /             \ (not OK. it will cause new-value store to be
//              /               X conditional on p0.new while R2 producer is
//             /                 \ on p0)
//            /                   \.
//     p.new store                 p.old NV store
// [if(p0.new)memw(R0+#0)=R2]    [if(p0)memw(R0+#0)=R2.new]
//            ^                  ^
//             \                /
//              \              /
//               \            /
//                 p.old store
//             [if (p0)memw(R0+#0)=R2]
//
// The above diagram shows the steps involoved in the conversion of a predicated
// store instruction to its .new predicated new-value form.
//
// The following set of instructions further explains the scenario where
// conditional new-value store becomes invalid when promoted to .new predicate
// form.
//
// { 1) if (p0) r0 = add(r1, r2)
//   2) p0 = cmp.eq(r3, #0) }
//
//   3) if (p0) memb(r1+#0) = r0  --> this instruction can't be grouped with
// the first two instructions because in instr 1, r0 is conditional on old value
// of p0 but its use in instr 3 is conditional on p0 modified by instr 2 which
// is not valid for new-value stores.
bool HexagonInstrInfo::
isConditionalStore (const MachineInstr* MI) const {
  const HexagonRegisterInfo& QRI = getRegisterInfo();
  switch (MI->getOpcode())
  {
    default: return false;
    case Hexagon::STrib_imm_cPt_V4 :
    case Hexagon::STrib_imm_cNotPt_V4 :
    case Hexagon::STrib_indexed_shl_cPt_V4 :
    case Hexagon::STrib_indexed_shl_cNotPt_V4 :
    case Hexagon::STrib_cPt :
    case Hexagon::STrib_cNotPt :
    case Hexagon::POST_STbri_cPt :
    case Hexagon::POST_STbri_cNotPt :
    case Hexagon::STrid_indexed_cPt :
    case Hexagon::STrid_indexed_cNotPt :
    case Hexagon::STrid_indexed_shl_cPt_V4 :
    case Hexagon::POST_STdri_cPt :
    case Hexagon::POST_STdri_cNotPt :
    case Hexagon::STrih_cPt :
    case Hexagon::STrih_cNotPt :
    case Hexagon::STrih_indexed_cPt :
    case Hexagon::STrih_indexed_cNotPt :
    case Hexagon::STrih_imm_cPt_V4 :
    case Hexagon::STrih_imm_cNotPt_V4 :
    case Hexagon::STrih_indexed_shl_cPt_V4 :
    case Hexagon::STrih_indexed_shl_cNotPt_V4 :
    case Hexagon::POST_SThri_cPt :
    case Hexagon::POST_SThri_cNotPt :
    case Hexagon::STriw_cPt :
    case Hexagon::STriw_cNotPt :
    case Hexagon::STriw_indexed_cPt :
    case Hexagon::STriw_indexed_cNotPt :
    case Hexagon::STriw_imm_cPt_V4 :
    case Hexagon::STriw_imm_cNotPt_V4 :
    case Hexagon::STriw_indexed_shl_cPt_V4 :
    case Hexagon::STriw_indexed_shl_cNotPt_V4 :
    case Hexagon::POST_STwri_cPt :
    case Hexagon::POST_STwri_cNotPt :
      return QRI.Subtarget.hasV4TOps();

    // V4 global address store before promoting to dot new.
    case Hexagon::STd_GP_cPt_V4 :
    case Hexagon::STd_GP_cNotPt_V4 :
    case Hexagon::STb_GP_cPt_V4 :
    case Hexagon::STb_GP_cNotPt_V4 :
    case Hexagon::STh_GP_cPt_V4 :
    case Hexagon::STh_GP_cNotPt_V4 :
    case Hexagon::STw_GP_cPt_V4 :
    case Hexagon::STw_GP_cNotPt_V4 :
      return QRI.Subtarget.hasV4TOps();

    // Predicated new value stores (i.e. if (p0) memw(..)=r0.new) are excluded
    // from the "Conditional Store" list. Because a predicated new value store
    // would NOT be promoted to a double dot new store. See diagram below:
    // This function returns yes for those stores that are predicated but not
    // yet promoted to predicate dot new instructions.
    //
    //                          +---------------------+
    //                    /-----| if (p0) memw(..)=r0 |---------\~
    //                   ||     +---------------------+         ||
    //          promote  ||       /\       /\                   ||  promote
    //                   ||      /||\     /||\                  ||
    //                  \||/    demote     ||                  \||/
    //                   \/       ||       ||                   \/
    //       +-------------------------+   ||   +-------------------------+
    //       | if (p0.new) memw(..)=r0 |   ||   | if (p0) memw(..)=r0.new |
    //       +-------------------------+   ||   +-------------------------+
    //                        ||           ||         ||
    //                        ||         demote      \||/
    //                      promote        ||         \/ NOT possible
    //                        ||           ||         /\~
    //                       \||/          ||        /||\~
    //                        \/           ||         ||
    //                      +-----------------------------+
    //                      | if (p0.new) memw(..)=r0.new |
    //                      +-----------------------------+
    //                           Double Dot New Store
    //
  }
}


bool HexagonInstrInfo::isNewValueJump(const MachineInstr *MI) const {
  if (isNewValue(MI) && isBranch(MI))
    return true;
  return false;
}

bool HexagonInstrInfo::isPostIncrement (const MachineInstr* MI) const {
  return (getAddrMode(MI) == HexagonII::PostInc);
}

bool HexagonInstrInfo::isNewValue(const MachineInstr* MI) const {
  const uint64_t F = MI->getDesc().TSFlags;
  return ((F >> HexagonII::NewValuePos) & HexagonII::NewValueMask);
}

// Returns true, if any one of the operands is a dot new
// insn, whether it is predicated dot new or register dot new.
bool HexagonInstrInfo::isDotNewInst (const MachineInstr* MI) const {
  return (isNewValueInst(MI) ||
     (isPredicated(MI) && isPredicatedNew(MI)));
}

// Returns the most basic instruction for the .new predicated instructions and
// new-value stores.
// For example, all of the following instructions will be converted back to the
// same instruction:
// 1) if (p0.new) memw(R0+#0) = R1.new  --->
// 2) if (p0) memw(R0+#0)= R1.new      -------> if (p0) memw(R0+#0) = R1
// 3) if (p0.new) memw(R0+#0) = R1      --->
//

int HexagonInstrInfo::GetDotOldOp(const int opc) const {
  int NewOp = opc;
  if (isPredicated(NewOp) && isPredicatedNew(NewOp)) { // Get predicate old form
    NewOp = Hexagon::getPredOldOpcode(NewOp);
    if (NewOp < 0)
      assert(0 && "Couldn't change predicate new instruction to its old form.");
  }

  if (isNewValueStore(NewOp)) { // Convert into non-new-value format
    NewOp = Hexagon::getNonNVStore(NewOp);
    if (NewOp < 0)
      assert(0 && "Couldn't change new-value store to its old form.");
  }
  return NewOp;
}

// Return the new value instruction for a given store.
int HexagonInstrInfo::GetDotNewOp(const MachineInstr* MI) const {
  int NVOpcode = Hexagon::getNewValueOpcode(MI->getOpcode());
  if (NVOpcode >= 0) // Valid new-value store instruction.
    return NVOpcode;

  switch (MI->getOpcode()) {
  default: llvm_unreachable("Unknown .new type");
  // store new value byte
  case Hexagon::STrib_shl_V4:
    return Hexagon::STrib_shl_nv_V4;

  case Hexagon::STrih_shl_V4:
    return Hexagon::STrih_shl_nv_V4;

  case Hexagon::STriw_f:
    return Hexagon::STriw_nv_V4;

  case Hexagon::STriw_indexed_f:
    return Hexagon::STriw_indexed_nv_V4;

  case Hexagon::STriw_shl_V4:
    return Hexagon::STriw_shl_nv_V4;

  }
  return 0;
}

// Return .new predicate version for an instruction.
int HexagonInstrInfo::GetDotNewPredOp(MachineInstr *MI,
                                      const MachineBranchProbabilityInfo
                                      *MBPI) const {

  int NewOpcode = Hexagon::getPredNewOpcode(MI->getOpcode());
  if (NewOpcode >= 0) // Valid predicate new instruction
    return NewOpcode;

  switch (MI->getOpcode()) {
  default: llvm_unreachable("Unknown .new type");
  // Condtional Jumps
  case Hexagon::JMP_t:
  case Hexagon::JMP_f:
    return getDotNewPredJumpOp(MI, MBPI);

  case Hexagon::JMPR_t:
    return Hexagon::JMPR_tnew_tV3;

  case Hexagon::JMPR_f:
    return Hexagon::JMPR_fnew_tV3;

  case Hexagon::JMPret_t:
    return Hexagon::JMPret_tnew_tV3;

  case Hexagon::JMPret_f:
    return Hexagon::JMPret_fnew_tV3;


  // Conditional combine
  case Hexagon::COMBINE_rr_cPt :
    return Hexagon::COMBINE_rr_cdnPt;
  case Hexagon::COMBINE_rr_cNotPt :
    return Hexagon::COMBINE_rr_cdnNotPt;
  }
}


unsigned HexagonInstrInfo::getAddrMode(const MachineInstr* MI) const {
  const uint64_t F = MI->getDesc().TSFlags;

  return((F >> HexagonII::AddrModePos) & HexagonII::AddrModeMask);
}

/// immediateExtend - Changes the instruction in place to one using an immediate
/// extender.
void HexagonInstrInfo::immediateExtend(MachineInstr *MI) const {
  assert((isExtendable(MI)||isConstExtended(MI)) &&
                               "Instruction must be extendable");
  // Find which operand is extendable.
  short ExtOpNum = getCExtOpNum(MI);
  MachineOperand &MO = MI->getOperand(ExtOpNum);
  // This needs to be something we understand.
  assert((MO.isMBB() || MO.isImm()) &&
         "Branch with unknown extendable field type");
  // Mark given operand as extended.
  MO.addTargetFlag(HexagonII::HMOTF_ConstExtended);
}

DFAPacketizer *HexagonInstrInfo::
CreateTargetScheduleState(const TargetMachine *TM,
                           const ScheduleDAG *DAG) const {
  const InstrItineraryData *II = TM->getInstrItineraryData();
  return TM->getSubtarget<HexagonGenSubtargetInfo>().createDFAPacketizer(II);
}

bool HexagonInstrInfo::isSchedulingBoundary(const MachineInstr *MI,
                                            const MachineBasicBlock *MBB,
                                            const MachineFunction &MF) const {
  // Debug info is never a scheduling boundary. It's necessary to be explicit
  // due to the special treatment of IT instructions below, otherwise a
  // dbg_value followed by an IT will result in the IT instruction being
  // considered a scheduling hazard, which is wrong. It should be the actual
  // instruction preceding the dbg_value instruction(s), just like it is
  // when debug info is not present.
  if (MI->isDebugValue())
    return false;

  // Terminators and labels can't be scheduled around.
  if (MI->getDesc().isTerminator() || MI->isPosition() || MI->isInlineAsm())
    return true;

  return false;
}

bool HexagonInstrInfo::isConstExtended(MachineInstr *MI) const {

  // Constant extenders are allowed only for V4 and above.
  if (!Subtarget.hasV4TOps())
    return false;

  const uint64_t F = MI->getDesc().TSFlags;
  unsigned isExtended = (F >> HexagonII::ExtendedPos) & HexagonII::ExtendedMask;
  if (isExtended) // Instruction must be extended.
    return true;

  unsigned isExtendable = (F >> HexagonII::ExtendablePos)
                          & HexagonII::ExtendableMask;
  if (!isExtendable)
    return false;

  short ExtOpNum = getCExtOpNum(MI);
  const MachineOperand &MO = MI->getOperand(ExtOpNum);
  // Use MO operand flags to determine if MO
  // has the HMOTF_ConstExtended flag set.
  if (MO.getTargetFlags() && HexagonII::HMOTF_ConstExtended)
    return true;
  // If this is a Machine BB address we are talking about, and it is
  // not marked as extended, say so.
  if (MO.isMBB())
    return false;

  // We could be using an instruction with an extendable immediate and shoehorn
  // a global address into it. If it is a global address it will be constant
  // extended. We do this for COMBINE.
  // We currently only handle isGlobal() because it is the only kind of
  // object we are going to end up with here for now.
  // In the future we probably should add isSymbol(), etc.
  if (MO.isGlobal() || MO.isSymbol())
    return true;

  // If the extendable operand is not 'Immediate' type, the instruction should
  // have 'isExtended' flag set.
  assert(MO.isImm() && "Extendable operand must be Immediate type");

  int MinValue = getMinValue(MI);
  int MaxValue = getMaxValue(MI);
  int ImmValue = MO.getImm();

  return (ImmValue < MinValue || ImmValue > MaxValue);
}

// Returns the opcode to use when converting MI, which is a conditional jump,
// into a conditional instruction which uses the .new value of the predicate.
// We also use branch probabilities to add a hint to the jump.
int
HexagonInstrInfo::getDotNewPredJumpOp(MachineInstr *MI,
                                  const
                                  MachineBranchProbabilityInfo *MBPI) const {

  // We assume that block can have at most two successors.
  bool taken = false;
  MachineBasicBlock *Src = MI->getParent();
  MachineOperand *BrTarget = &MI->getOperand(1);
  MachineBasicBlock *Dst = BrTarget->getMBB();

  const BranchProbability Prediction = MBPI->getEdgeProbability(Src, Dst);
  if (Prediction >= BranchProbability(1,2))
    taken = true;

  switch (MI->getOpcode()) {
  case Hexagon::JMP_t:
    return taken ? Hexagon::JMP_tnew_t : Hexagon::JMP_tnew_nt;
  case Hexagon::JMP_f:
    return taken ? Hexagon::JMP_fnew_t : Hexagon::JMP_fnew_nt;

  default:
    llvm_unreachable("Unexpected jump instruction.");
  }
}
// Returns true if a particular operand is extendable for an instruction.
bool HexagonInstrInfo::isOperandExtended(const MachineInstr *MI,
                                         unsigned short OperandNum) const {
  // Constant extenders are allowed only for V4 and above.
  if (!Subtarget.hasV4TOps())
    return false;

  const uint64_t F = MI->getDesc().TSFlags;

  return ((F >> HexagonII::ExtendableOpPos) & HexagonII::ExtendableOpMask)
          == OperandNum;
}

// Returns Operand Index for the constant extended instruction.
unsigned short HexagonInstrInfo::getCExtOpNum(const MachineInstr *MI) const {
  const uint64_t F = MI->getDesc().TSFlags;
  return ((F >> HexagonII::ExtendableOpPos) & HexagonII::ExtendableOpMask);
}

// Returns the min value that doesn't need to be extended.
int HexagonInstrInfo::getMinValue(const MachineInstr *MI) const {
  const uint64_t F = MI->getDesc().TSFlags;
  unsigned isSigned = (F >> HexagonII::ExtentSignedPos)
                    & HexagonII::ExtentSignedMask;
  unsigned bits =  (F >> HexagonII::ExtentBitsPos)
                    & HexagonII::ExtentBitsMask;

  if (isSigned) // if value is signed
    return -1 << (bits - 1);
  else
    return 0;
}

// Returns the max value that doesn't need to be extended.
int HexagonInstrInfo::getMaxValue(const MachineInstr *MI) const {
  const uint64_t F = MI->getDesc().TSFlags;
  unsigned isSigned = (F >> HexagonII::ExtentSignedPos)
                    & HexagonII::ExtentSignedMask;
  unsigned bits =  (F >> HexagonII::ExtentBitsPos)
                    & HexagonII::ExtentBitsMask;

  if (isSigned) // if value is signed
    return ~(-1 << (bits - 1));
  else
    return ~(-1 << bits);
}

// Returns true if an instruction can be converted into a non-extended
// equivalent instruction.
bool HexagonInstrInfo::NonExtEquivalentExists (const MachineInstr *MI) const {

  short NonExtOpcode;
  // Check if the instruction has a register form that uses register in place
  // of the extended operand, if so return that as the non-extended form.
  if (Hexagon::getRegForm(MI->getOpcode()) >= 0)
    return true;

  if (MI->getDesc().mayLoad() || MI->getDesc().mayStore()) {
    // Check addressing mode and retrieve non-ext equivalent instruction.

    switch (getAddrMode(MI)) {
    case HexagonII::Absolute :
      // Load/store with absolute addressing mode can be converted into
      // base+offset mode.
      NonExtOpcode = Hexagon::getBasedWithImmOffset(MI->getOpcode());
      break;
    case HexagonII::BaseImmOffset :
      // Load/store with base+offset addressing mode can be converted into
      // base+register offset addressing mode. However left shift operand should
      // be set to 0.
      NonExtOpcode = Hexagon::getBaseWithRegOffset(MI->getOpcode());
      break;
    default:
      return false;
    }
    if (NonExtOpcode < 0)
      return false;
    return true;
  }
  return false;
}

// Returns opcode of the non-extended equivalent instruction.
short HexagonInstrInfo::getNonExtOpcode (const MachineInstr *MI) const {

  // Check if the instruction has a register form that uses register in place
  // of the extended operand, if so return that as the non-extended form.
  short NonExtOpcode = Hexagon::getRegForm(MI->getOpcode());
    if (NonExtOpcode >= 0)
      return NonExtOpcode;

  if (MI->getDesc().mayLoad() || MI->getDesc().mayStore()) {
    // Check addressing mode and retrieve non-ext equivalent instruction.
    switch (getAddrMode(MI)) {
    case HexagonII::Absolute :
      return Hexagon::getBasedWithImmOffset(MI->getOpcode());
    case HexagonII::BaseImmOffset :
      return Hexagon::getBaseWithRegOffset(MI->getOpcode());
    default:
      return -1;
    }
  }
  return -1;
}

bool HexagonInstrInfo::PredOpcodeHasJMP_c(Opcode_t Opcode) const {
  return (Opcode == Hexagon::JMP_t) ||
         (Opcode == Hexagon::JMP_f) ||
         (Opcode == Hexagon::JMP_tnew_t) ||
         (Opcode == Hexagon::JMP_fnew_t) ||
         (Opcode == Hexagon::JMP_tnew_nt) ||
         (Opcode == Hexagon::JMP_fnew_nt);
}

bool HexagonInstrInfo::PredOpcodeHasNot(Opcode_t Opcode) const {
  return (Opcode == Hexagon::JMP_f) ||
         (Opcode == Hexagon::JMP_fnew_t) ||
         (Opcode == Hexagon::JMP_fnew_nt);
}
