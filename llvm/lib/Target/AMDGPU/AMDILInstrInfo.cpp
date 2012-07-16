//===- AMDILInstrInfo.cpp - AMDIL Instruction Information -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//==-----------------------------------------------------------------------===//
//
// This file contains the AMDIL implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "AMDILInstrInfo.h"
#include "AMDIL.h"
#include "AMDILISelLowering.h"
#include "AMDILUtilityFunctions.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/PseudoSourceValue.h"
#include "llvm/Instructions.h"

#define GET_INSTRINFO_CTOR
#include "AMDGPUGenInstrInfo.inc"

using namespace llvm;

AMDILInstrInfo::AMDILInstrInfo(TargetMachine &tm)
  : AMDILGenInstrInfo(),
    RI(tm, *this) {
}

const AMDILRegisterInfo &AMDILInstrInfo::getRegisterInfo() const {
  return RI;
}

bool AMDILInstrInfo::isCoalescableExtInstr(const MachineInstr &MI,
                                           unsigned &SrcReg, unsigned &DstReg,
                                           unsigned &SubIdx) const {
// TODO: Implement this function
  return false;
}

unsigned AMDILInstrInfo::isLoadFromStackSlot(const MachineInstr *MI,
                                             int &FrameIndex) const {
// TODO: Implement this function
  return 0;
}

unsigned AMDILInstrInfo::isLoadFromStackSlotPostFE(const MachineInstr *MI,
                                                   int &FrameIndex) const {
// TODO: Implement this function
  return 0;
}

bool AMDILInstrInfo::hasLoadFromStackSlot(const MachineInstr *MI,
                                          const MachineMemOperand *&MMO,
                                          int &FrameIndex) const {
// TODO: Implement this function
  return false;
}
unsigned AMDILInstrInfo::isStoreFromStackSlot(const MachineInstr *MI,
                                              int &FrameIndex) const {
// TODO: Implement this function
  return 0;
}
unsigned AMDILInstrInfo::isStoreFromStackSlotPostFE(const MachineInstr *MI,
                                                    int &FrameIndex) const {
// TODO: Implement this function
  return 0;
}
bool AMDILInstrInfo::hasStoreFromStackSlot(const MachineInstr *MI,
                                           const MachineMemOperand *&MMO,
                                           int &FrameIndex) const {
// TODO: Implement this function
  return false;
}

MachineInstr *
AMDILInstrInfo::convertToThreeAddress(MachineFunction::iterator &MFI,
                                      MachineBasicBlock::iterator &MBBI,
                                      LiveVariables *LV) const {
// TODO: Implement this function
  return NULL;
}
bool AMDILInstrInfo::getNextBranchInstr(MachineBasicBlock::iterator &iter,
                                        MachineBasicBlock &MBB) const {
  while (iter != MBB.end()) {
    switch (iter->getOpcode()) {
    default:
      break;
      ExpandCaseToAllScalarTypes(AMDGPU::BRANCH_COND);
    case AMDGPU::BRANCH:
      return true;
    };
    ++iter;
  }
  return false;
}

bool AMDILInstrInfo::AnalyzeBranch(MachineBasicBlock &MBB,
                                   MachineBasicBlock *&TBB,
                                   MachineBasicBlock *&FBB,
                                   SmallVectorImpl<MachineOperand> &Cond,
                                   bool AllowModify) const {
  bool retVal = true;
  return retVal;
  MachineBasicBlock::iterator iter = MBB.begin();
  if (!getNextBranchInstr(iter, MBB)) {
    retVal = false;
  } else {
    MachineInstr *firstBranch = iter;
    if (!getNextBranchInstr(++iter, MBB)) {
      if (firstBranch->getOpcode() == AMDGPU::BRANCH) {
        TBB = firstBranch->getOperand(0).getMBB();
        firstBranch->eraseFromParent();
        retVal = false;
      } else {
        TBB = firstBranch->getOperand(0).getMBB();
        FBB = *(++MBB.succ_begin());
        if (FBB == TBB) {
          FBB = *(MBB.succ_begin());
        }
        Cond.push_back(firstBranch->getOperand(1));
        retVal = false;
      }
    } else {
      MachineInstr *secondBranch = iter;
      if (!getNextBranchInstr(++iter, MBB)) {
        if (secondBranch->getOpcode() == AMDGPU::BRANCH) {
          TBB = firstBranch->getOperand(0).getMBB();
          Cond.push_back(firstBranch->getOperand(1));
          FBB = secondBranch->getOperand(0).getMBB();
          secondBranch->eraseFromParent();
          retVal = false;
        } else {
          assert(0 && "Should not have two consecutive conditional branches");
        }
      } else {
        MBB.getParent()->viewCFG();
        assert(0 && "Should not have three branch instructions in"
               " a single basic block");
        retVal = false;
      }
    }
  }
  return retVal;
}

unsigned int AMDILInstrInfo::getBranchInstr(const MachineOperand &op) const {
  const MachineInstr *MI = op.getParent();
  
  switch (MI->getDesc().OpInfo->RegClass) {
  default: // FIXME: fallthrough??
  case AMDGPU::GPRI32RegClassID: return AMDGPU::BRANCH_COND_i32;
  case AMDGPU::GPRF32RegClassID: return AMDGPU::BRANCH_COND_f32;
  };
}

unsigned int
AMDILInstrInfo::InsertBranch(MachineBasicBlock &MBB,
                             MachineBasicBlock *TBB,
                             MachineBasicBlock *FBB,
                             const SmallVectorImpl<MachineOperand> &Cond,
                             DebugLoc DL) const
{
  assert(TBB && "InsertBranch must not be told to insert a fallthrough");
  for (unsigned int x = 0; x < Cond.size(); ++x) {
    Cond[x].getParent()->dump();
  }
  if (FBB == 0) {
    if (Cond.empty()) {
      BuildMI(&MBB, DL, get(AMDGPU::BRANCH)).addMBB(TBB);
    } else {
      BuildMI(&MBB, DL, get(getBranchInstr(Cond[0])))
        .addMBB(TBB).addReg(Cond[0].getReg());
    }
    return 1;
  } else {
    BuildMI(&MBB, DL, get(getBranchInstr(Cond[0])))
      .addMBB(TBB).addReg(Cond[0].getReg());
    BuildMI(&MBB, DL, get(AMDGPU::BRANCH)).addMBB(FBB);
  }
  assert(0 && "Inserting two branches not supported");
  return 0;
}

unsigned int AMDILInstrInfo::RemoveBranch(MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin()) {
    return 0;
  }
  --I;
  switch (I->getOpcode()) {
  default:
    return 0;
    ExpandCaseToAllScalarTypes(AMDGPU::BRANCH_COND);
  case AMDGPU::BRANCH:
    I->eraseFromParent();
    break;
  }
  I = MBB.end();
  
  if (I == MBB.begin()) {
    return 1;
  }
  --I;
  switch (I->getOpcode()) {
    // FIXME: only one case??
  default:
    return 1;
    ExpandCaseToAllScalarTypes(AMDGPU::BRANCH_COND);
    I->eraseFromParent();
    break;
  }
  return 2;
}

MachineBasicBlock::iterator skipFlowControl(MachineBasicBlock *MBB) {
  MachineBasicBlock::iterator tmp = MBB->end();
  if (!MBB->size()) {
    return MBB->end();
  }
  while (--tmp) {
    if (tmp->getOpcode() == AMDGPU::ENDLOOP
        || tmp->getOpcode() == AMDGPU::ENDIF
        || tmp->getOpcode() == AMDGPU::ELSE) {
      if (tmp == MBB->begin()) {
        return tmp;
      } else {
        continue;
      }
    }  else {
      return ++tmp;
    }
  }
  return MBB->end();
}

void
AMDILInstrInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                    MachineBasicBlock::iterator MI,
                                    unsigned SrcReg, bool isKill,
                                    int FrameIndex,
                                    const TargetRegisterClass *RC,
                                    const TargetRegisterInfo *TRI) const {
  unsigned int Opc = 0;
  // MachineInstr *curMI = MI;
  MachineFunction &MF = *(MBB.getParent());
  MachineFrameInfo &MFI = *MF.getFrameInfo();
  
  DebugLoc DL;
  switch (RC->getID()) {
  case AMDGPU::GPRF32RegClassID:
    Opc = AMDGPU::PRIVATESTORE_f32;
    break;
  case AMDGPU::GPRI32RegClassID:
    Opc = AMDGPU::PRIVATESTORE_i32;
    break;
  }
  if (MI != MBB.end()) DL = MI->getDebugLoc();
  MachineMemOperand *MMO =
   new MachineMemOperand(
        MachinePointerInfo::getFixedStack(FrameIndex),
                          MachineMemOperand::MOLoad,
                          MFI.getObjectSize(FrameIndex),
                          MFI.getObjectAlignment(FrameIndex));
  if (MI != MBB.end()) {
    DL = MI->getDebugLoc();
  }
  BuildMI(MBB, MI, DL, get(Opc))
    .addReg(SrcReg, getKillRegState(isKill))
    .addFrameIndex(FrameIndex)
    .addMemOperand(MMO)
    .addImm(0);
}

void
AMDILInstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                     MachineBasicBlock::iterator MI,
                                     unsigned DestReg, int FrameIndex,
                                     const TargetRegisterClass *RC,
                                     const TargetRegisterInfo *TRI) const {
  unsigned int Opc = 0;
  MachineFunction &MF = *(MBB.getParent());
  MachineFrameInfo &MFI = *MF.getFrameInfo();
  DebugLoc DL;
  switch (RC->getID()) {
  case AMDGPU::GPRF32RegClassID:
    Opc = AMDGPU::PRIVATELOAD_f32;
    break;
  case AMDGPU::GPRI32RegClassID:
    Opc = AMDGPU::PRIVATELOAD_i32;
    break;
  }

  MachineMemOperand *MMO =
    new MachineMemOperand(
        MachinePointerInfo::getFixedStack(FrameIndex),
                          MachineMemOperand::MOLoad,
                          MFI.getObjectSize(FrameIndex),
                          MFI.getObjectAlignment(FrameIndex));
  if (MI != MBB.end()) {
    DL = MI->getDebugLoc();
  }
  BuildMI(MBB, MI, DL, get(Opc))
    .addReg(DestReg, RegState::Define)
    .addFrameIndex(FrameIndex)
    .addMemOperand(MMO)
    .addImm(0);
}
MachineInstr *
AMDILInstrInfo::foldMemoryOperandImpl(MachineFunction &MF,
                                      MachineInstr *MI,
                                      const SmallVectorImpl<unsigned> &Ops,
                                      int FrameIndex) const {
// TODO: Implement this function
  return 0;
}
MachineInstr*
AMDILInstrInfo::foldMemoryOperandImpl(MachineFunction &MF,
                                      MachineInstr *MI,
                                      const SmallVectorImpl<unsigned> &Ops,
                                      MachineInstr *LoadMI) const {
  // TODO: Implement this function
  return 0;
}
bool
AMDILInstrInfo::canFoldMemoryOperand(const MachineInstr *MI,
                                     const SmallVectorImpl<unsigned> &Ops) const
{
  // TODO: Implement this function
  return false;
}
bool
AMDILInstrInfo::unfoldMemoryOperand(MachineFunction &MF, MachineInstr *MI,
                                 unsigned Reg, bool UnfoldLoad,
                                 bool UnfoldStore,
                                 SmallVectorImpl<MachineInstr*> &NewMIs) const {
  // TODO: Implement this function
  return false;
}

bool
AMDILInstrInfo::unfoldMemoryOperand(SelectionDAG &DAG, SDNode *N,
                                    SmallVectorImpl<SDNode*> &NewNodes) const {
  // TODO: Implement this function
  return false;
}

unsigned
AMDILInstrInfo::getOpcodeAfterMemoryUnfold(unsigned Opc,
                                           bool UnfoldLoad, bool UnfoldStore,
                                           unsigned *LoadRegIndex) const {
  // TODO: Implement this function
  return 0;
}

bool AMDILInstrInfo::shouldScheduleLoadsNear(SDNode *Load1, SDNode *Load2,
                                             int64_t Offset1, int64_t Offset2,
                                             unsigned NumLoads) const {
  assert(Offset2 > Offset1
         && "Second offset should be larger than first offset!");
  // If we have less than 16 loads in a row, and the offsets are within 16,
  // then schedule together.
  // TODO: Make the loads schedule near if it fits in a cacheline
  return (NumLoads < 16 && (Offset2 - Offset1) < 16);
}

bool
AMDILInstrInfo::ReverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond)
  const {
  // TODO: Implement this function
  return true;
}
void AMDILInstrInfo::insertNoop(MachineBasicBlock &MBB,
                                MachineBasicBlock::iterator MI) const {
  // TODO: Implement this function
}

bool AMDILInstrInfo::isPredicated(const MachineInstr *MI) const {
  // TODO: Implement this function
  return false;
}
bool
AMDILInstrInfo::SubsumesPredicate(const SmallVectorImpl<MachineOperand> &Pred1,
                                  const SmallVectorImpl<MachineOperand> &Pred2)
  const {
  // TODO: Implement this function
  return false;
}

bool AMDILInstrInfo::DefinesPredicate(MachineInstr *MI,
                                      std::vector<MachineOperand> &Pred) const {
  // TODO: Implement this function
  return false;
}

bool AMDILInstrInfo::isPredicable(MachineInstr *MI) const {
  // TODO: Implement this function
  return MI->getDesc().isPredicable();
}

bool
AMDILInstrInfo::isSafeToMoveRegClassDefs(const TargetRegisterClass *RC) const {
  // TODO: Implement this function
  return true;
}

bool AMDILInstrInfo::isLoadInst(MachineInstr *MI) const {
  if (strstr(getName(MI->getOpcode()), "LOADCONST")) {
    return false;
  }
  return strstr(getName(MI->getOpcode()), "LOAD");
}

bool AMDILInstrInfo::isSWSExtLoadInst(MachineInstr *MI) const
{
  return false;
}

bool AMDILInstrInfo::isExtLoadInst(MachineInstr *MI) const {
  return strstr(getName(MI->getOpcode()), "EXTLOAD");
}

bool AMDILInstrInfo::isSExtLoadInst(MachineInstr *MI) const {
  return strstr(getName(MI->getOpcode()), "SEXTLOAD");
}

bool AMDILInstrInfo::isAExtLoadInst(MachineInstr *MI) const {
  return strstr(getName(MI->getOpcode()), "AEXTLOAD");
}

bool AMDILInstrInfo::isZExtLoadInst(MachineInstr *MI) const {
  return strstr(getName(MI->getOpcode()), "ZEXTLOAD");
}

bool AMDILInstrInfo::isStoreInst(MachineInstr *MI) const {
  return strstr(getName(MI->getOpcode()), "STORE");
}

bool AMDILInstrInfo::isTruncStoreInst(MachineInstr *MI) const {
  return strstr(getName(MI->getOpcode()), "TRUNCSTORE");
}

bool AMDILInstrInfo::isAtomicInst(MachineInstr *MI) const {
  return strstr(getName(MI->getOpcode()), "ATOM");
}

bool AMDILInstrInfo::isVolatileInst(MachineInstr *MI) const {
  if (!MI->memoperands_empty()) {
    for (MachineInstr::mmo_iterator mob = MI->memoperands_begin(),
        moe = MI->memoperands_end(); mob != moe; ++mob) {
      // If there is a volatile mem operand, this is a volatile instruction.
      if ((*mob)->isVolatile()) {
        return true;
      }
    }
  }
  return false;
}
bool AMDILInstrInfo::isGlobalInst(llvm::MachineInstr *MI) const
{
  return strstr(getName(MI->getOpcode()), "GLOBAL");
}
bool AMDILInstrInfo::isPrivateInst(llvm::MachineInstr *MI) const
{
  return strstr(getName(MI->getOpcode()), "PRIVATE");
}
bool AMDILInstrInfo::isConstantInst(llvm::MachineInstr *MI) const
{
  return strstr(getName(MI->getOpcode()), "CONSTANT")
    || strstr(getName(MI->getOpcode()), "CPOOL");
}
bool AMDILInstrInfo::isRegionInst(llvm::MachineInstr *MI) const
{
  return strstr(getName(MI->getOpcode()), "REGION");
}
bool AMDILInstrInfo::isLocalInst(llvm::MachineInstr *MI) const
{
  return strstr(getName(MI->getOpcode()), "LOCAL");
}
bool AMDILInstrInfo::isImageInst(llvm::MachineInstr *MI) const
{
  return strstr(getName(MI->getOpcode()), "IMAGE");
}
bool AMDILInstrInfo::isAppendInst(llvm::MachineInstr *MI) const
{
  return strstr(getName(MI->getOpcode()), "APPEND");
}
bool AMDILInstrInfo::isRegionAtomic(llvm::MachineInstr *MI) const
{
  return strstr(getName(MI->getOpcode()), "ATOM_R");
}
bool AMDILInstrInfo::isLocalAtomic(llvm::MachineInstr *MI) const
{
  return strstr(getName(MI->getOpcode()), "ATOM_L");
}
bool AMDILInstrInfo::isGlobalAtomic(llvm::MachineInstr *MI) const
{
  return strstr(getName(MI->getOpcode()), "ATOM_G")
    || isArenaAtomic(MI);
}
bool AMDILInstrInfo::isArenaAtomic(llvm::MachineInstr *MI) const
{
  return strstr(getName(MI->getOpcode()), "ATOM_A");
}
