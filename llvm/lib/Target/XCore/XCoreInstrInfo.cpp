//===- XCoreInstrInfo.cpp - XCore Instruction Information -------*- C++ -*-===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This file contains the XCore implementation of the TargetInstrInfo class.
//
//===----------------------------------------------------------------------===//

#include "XCoreMachineFunctionInfo.h"
#include "XCoreInstrInfo.h"
#include "XCore.h"
#include "llvm/MC/MCContext.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/CodeGen/MachineFrameInfo.h"
#include "llvm/CodeGen/MachineLocation.h"
#include "XCoreGenInstrInfo.inc"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"

namespace llvm {
namespace XCore {

  // XCore Condition Codes
  enum CondCode {
    COND_TRUE,
    COND_FALSE,
    COND_INVALID
  };
}
}

using namespace llvm;

XCoreInstrInfo::XCoreInstrInfo()
  : TargetInstrInfoImpl(XCoreInsts, array_lengthof(XCoreInsts)),
    RI(*this) {
}

static bool isZeroImm(const MachineOperand &op) {
  return op.isImm() && op.getImm() == 0;
}

/// Return true if the instruction is a register to register move and
/// leave the source and dest operands in the passed parameters.
///
bool XCoreInstrInfo::isMoveInstr(const MachineInstr &MI,
                                 unsigned &SrcReg, unsigned &DstReg,
                                 unsigned &SrcSR, unsigned &DstSR) const {
  SrcSR = DstSR = 0; // No sub-registers.

  // We look for 4 kinds of patterns here:
  // add dst, src, 0
  // sub dst, src, 0
  // or dst, src, src
  // and dst, src, src
  if ((MI.getOpcode() == XCore::ADD_2rus || MI.getOpcode() == XCore::SUB_2rus)
      && isZeroImm(MI.getOperand(2))) {
    DstReg = MI.getOperand(0).getReg();
    SrcReg = MI.getOperand(1).getReg();
    return true;
  } else if ((MI.getOpcode() == XCore::OR_3r || MI.getOpcode() == XCore::AND_3r)
      && MI.getOperand(1).getReg() == MI.getOperand(2).getReg()) {
    DstReg = MI.getOperand(0).getReg();
    SrcReg = MI.getOperand(1).getReg();
    return true;
  }
  return false;
}

/// isLoadFromStackSlot - If the specified machine instruction is a direct
/// load from a stack slot, return the virtual or physical register number of
/// the destination along with the FrameIndex of the loaded stack slot.  If
/// not, return 0.  This predicate must return 0 if the instruction has
/// any side effects other than loading from the stack slot.
unsigned
XCoreInstrInfo::isLoadFromStackSlot(const MachineInstr *MI, int &FrameIndex) const{
  int Opcode = MI->getOpcode();
  if (Opcode == XCore::LDWFI) 
  {
    if ((MI->getOperand(1).isFI()) && // is a stack slot
        (MI->getOperand(2).isImm()) &&  // the imm is zero
        (isZeroImm(MI->getOperand(2)))) 
    {
      FrameIndex = MI->getOperand(1).getIndex();
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
unsigned
XCoreInstrInfo::isStoreToStackSlot(const MachineInstr *MI,
                                   int &FrameIndex) const {
  int Opcode = MI->getOpcode();
  if (Opcode == XCore::STWFI)
  {
    if ((MI->getOperand(1).isFI()) && // is a stack slot
        (MI->getOperand(2).isImm()) &&  // the imm is zero
        (isZeroImm(MI->getOperand(2))))
    {
      FrameIndex = MI->getOperand(1).getIndex();
      return MI->getOperand(0).getReg();
    }
  }
  return 0;
}

//===----------------------------------------------------------------------===//
// Branch Analysis
//===----------------------------------------------------------------------===//

static inline bool IsBRU(unsigned BrOpc) {
  return BrOpc == XCore::BRFU_u6
      || BrOpc == XCore::BRFU_lu6
      || BrOpc == XCore::BRBU_u6
      || BrOpc == XCore::BRBU_lu6;
}

static inline bool IsBRT(unsigned BrOpc) {
  return BrOpc == XCore::BRFT_ru6
      || BrOpc == XCore::BRFT_lru6
      || BrOpc == XCore::BRBT_ru6
      || BrOpc == XCore::BRBT_lru6;
}

static inline bool IsBRF(unsigned BrOpc) {
  return BrOpc == XCore::BRFF_ru6
      || BrOpc == XCore::BRFF_lru6
      || BrOpc == XCore::BRBF_ru6
      || BrOpc == XCore::BRBF_lru6;
}

static inline bool IsCondBranch(unsigned BrOpc) {
  return IsBRF(BrOpc) || IsBRT(BrOpc);
}

static inline bool IsBR_JT(unsigned BrOpc) {
  return BrOpc == XCore::BR_JT
      || BrOpc == XCore::BR_JT32;
}

/// GetCondFromBranchOpc - Return the XCore CC that matches 
/// the correspondent Branch instruction opcode.
static XCore::CondCode GetCondFromBranchOpc(unsigned BrOpc) 
{
  if (IsBRT(BrOpc)) {
    return XCore::COND_TRUE;
  } else if (IsBRF(BrOpc)) {
    return XCore::COND_FALSE;
  } else {
    return XCore::COND_INVALID;
  }
}

/// GetCondBranchFromCond - Return the Branch instruction
/// opcode that matches the cc.
static inline unsigned GetCondBranchFromCond(XCore::CondCode CC) 
{
  switch (CC) {
  default: llvm_unreachable("Illegal condition code!");
  case XCore::COND_TRUE   : return XCore::BRFT_lru6;
  case XCore::COND_FALSE  : return XCore::BRFF_lru6;
  }
}

/// GetOppositeBranchCondition - Return the inverse of the specified 
/// condition, e.g. turning COND_E to COND_NE.
static inline XCore::CondCode GetOppositeBranchCondition(XCore::CondCode CC)
{
  switch (CC) {
  default: llvm_unreachable("Illegal condition code!");
  case XCore::COND_TRUE   : return XCore::COND_FALSE;
  case XCore::COND_FALSE  : return XCore::COND_TRUE;
  }
}

/// AnalyzeBranch - Analyze the branching code at the end of MBB, returning
/// true if it cannot be understood (e.g. it's a switch dispatch or isn't
/// implemented for a target).  Upon success, this returns false and returns
/// with the following information in various cases:
///
/// 1. If this block ends with no branches (it just falls through to its succ)
///    just return false, leaving TBB/FBB null.
/// 2. If this block ends with only an unconditional branch, it sets TBB to be
///    the destination block.
/// 3. If this block ends with an conditional branch and it falls through to
///    an successor block, it sets TBB to be the branch destination block and a
///    list of operands that evaluate the condition. These
///    operands can be passed to other TargetInstrInfo methods to create new
///    branches.
/// 4. If this block ends with an conditional branch and an unconditional
///    block, it returns the 'true' destination in TBB, the 'false' destination
///    in FBB, and a list of operands that evaluate the condition. These
///    operands can be passed to other TargetInstrInfo methods to create new
///    branches.
///
/// Note that RemoveBranch and InsertBranch must be implemented to support
/// cases where this method returns success.
///
bool
XCoreInstrInfo::AnalyzeBranch(MachineBasicBlock &MBB, MachineBasicBlock *&TBB,
                              MachineBasicBlock *&FBB,
                              SmallVectorImpl<MachineOperand> &Cond,
                              bool AllowModify) const {
  // If the block has no terminators, it just falls into the block after it.
  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin())
    return false;
  --I;
  while (I->isDebugValue()) {
    if (I == MBB.begin())
      return false;
    --I;
  }
  if (!isUnpredicatedTerminator(I))
    return false;

  // Get the last instruction in the block.
  MachineInstr *LastInst = I;
  
  // If there is only one terminator instruction, process it.
  if (I == MBB.begin() || !isUnpredicatedTerminator(--I)) {
    if (IsBRU(LastInst->getOpcode())) {
      TBB = LastInst->getOperand(0).getMBB();
      return false;
    }
    
    XCore::CondCode BranchCode = GetCondFromBranchOpc(LastInst->getOpcode());
    if (BranchCode == XCore::COND_INVALID)
      return true;  // Can't handle indirect branch.
    
    // Conditional branch
    // Block ends with fall-through condbranch.

    TBB = LastInst->getOperand(1).getMBB();
    Cond.push_back(MachineOperand::CreateImm(BranchCode));
    Cond.push_back(LastInst->getOperand(0));
    return false;
  }
  
  // Get the instruction before it if it's a terminator.
  MachineInstr *SecondLastInst = I;

  // If there are three terminators, we don't know what sort of block this is.
  if (SecondLastInst && I != MBB.begin() &&
      isUnpredicatedTerminator(--I))
    return true;
  
  unsigned SecondLastOpc    = SecondLastInst->getOpcode();
  XCore::CondCode BranchCode = GetCondFromBranchOpc(SecondLastOpc);
  
  // If the block ends with conditional branch followed by unconditional,
  // handle it.
  if (BranchCode != XCore::COND_INVALID
    && IsBRU(LastInst->getOpcode())) {

    TBB = SecondLastInst->getOperand(1).getMBB();
    Cond.push_back(MachineOperand::CreateImm(BranchCode));
    Cond.push_back(SecondLastInst->getOperand(0));

    FBB = LastInst->getOperand(0).getMBB();
    return false;
  }
  
  // If the block ends with two unconditional branches, handle it.  The second
  // one is not executed, so remove it.
  if (IsBRU(SecondLastInst->getOpcode()) && 
      IsBRU(LastInst->getOpcode())) {
    TBB = SecondLastInst->getOperand(0).getMBB();
    I = LastInst;
    if (AllowModify)
      I->eraseFromParent();
    return false;
  }

  // Likewise if it ends with a branch table followed by an unconditional branch.
  if (IsBR_JT(SecondLastInst->getOpcode()) && IsBRU(LastInst->getOpcode())) {
    I = LastInst;
    if (AllowModify)
      I->eraseFromParent();
    return true;
  }

  // Otherwise, can't handle this.
  return true;
}

unsigned
XCoreInstrInfo::InsertBranch(MachineBasicBlock &MBB,MachineBasicBlock *TBB,
                             MachineBasicBlock *FBB,
                             const SmallVectorImpl<MachineOperand> &Cond)const{
  // FIXME there should probably be a DebugLoc argument here
  DebugLoc dl;
  // Shouldn't be a fall through.
  assert(TBB && "InsertBranch must not be told to insert a fallthrough");
  assert((Cond.size() == 2 || Cond.size() == 0) &&
         "Unexpected number of components!");
  
  if (FBB == 0) { // One way branch.
    if (Cond.empty()) {
      // Unconditional branch
      BuildMI(&MBB, dl, get(XCore::BRFU_lu6)).addMBB(TBB);
    } else {
      // Conditional branch.
      unsigned Opc = GetCondBranchFromCond((XCore::CondCode)Cond[0].getImm());
      BuildMI(&MBB, dl, get(Opc)).addReg(Cond[1].getReg())
                             .addMBB(TBB);
    }
    return 1;
  }
  
  // Two-way Conditional branch.
  assert(Cond.size() == 2 && "Unexpected number of components!");
  unsigned Opc = GetCondBranchFromCond((XCore::CondCode)Cond[0].getImm());
  BuildMI(&MBB, dl, get(Opc)).addReg(Cond[1].getReg())
                         .addMBB(TBB);
  BuildMI(&MBB, dl, get(XCore::BRFU_lu6)).addMBB(FBB);
  return 2;
}

unsigned
XCoreInstrInfo::RemoveBranch(MachineBasicBlock &MBB) const {
  MachineBasicBlock::iterator I = MBB.end();
  if (I == MBB.begin()) return 0;
  --I;
  while (I->isDebugValue()) {
    if (I == MBB.begin())
      return 0;
    --I;
  }
  if (!IsBRU(I->getOpcode()) && !IsCondBranch(I->getOpcode()))
    return 0;
  
  // Remove the branch.
  I->eraseFromParent();
  
  I = MBB.end();

  if (I == MBB.begin()) return 1;
  --I;
  if (!IsCondBranch(I->getOpcode()))
    return 1;
  
  // Remove the branch.
  I->eraseFromParent();
  return 2;
}

bool XCoreInstrInfo::copyRegToReg(MachineBasicBlock &MBB,
                                  MachineBasicBlock::iterator I,
                                  unsigned DestReg, unsigned SrcReg,
                                  const TargetRegisterClass *DestRC,
                                  const TargetRegisterClass *SrcRC,
                                  DebugLoc DL) const {

  if (DestRC == SrcRC) {
    if (DestRC == XCore::GRRegsRegisterClass) {
      BuildMI(MBB, I, DL, get(XCore::ADD_2rus), DestReg)
        .addReg(SrcReg)
        .addImm(0);
      return true;
    } else {
      return false;
    }
  }
  
  if (SrcRC == XCore::RRegsRegisterClass && SrcReg == XCore::SP &&
    DestRC == XCore::GRRegsRegisterClass) {
    BuildMI(MBB, I, DL, get(XCore::LDAWSP_ru6), DestReg)
      .addImm(0);
    return true;
  }
  if (DestRC == XCore::RRegsRegisterClass && DestReg == XCore::SP &&
    SrcRC == XCore::GRRegsRegisterClass) {
    BuildMI(MBB, I, DL, get(XCore::SETSP_1r))
      .addReg(SrcReg);
    return true;
  }
  return false;
}

void XCoreInstrInfo::storeRegToStackSlot(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator I,
                                         unsigned SrcReg, bool isKill,
                                         int FrameIndex,
                                         const TargetRegisterClass *RC,
                                         const TargetRegisterInfo *TRI) const
{
  DebugLoc DL;
  if (I != MBB.end()) DL = I->getDebugLoc();
  BuildMI(MBB, I, DL, get(XCore::STWFI))
    .addReg(SrcReg, getKillRegState(isKill))
    .addFrameIndex(FrameIndex)
    .addImm(0);
}

void XCoreInstrInfo::loadRegFromStackSlot(MachineBasicBlock &MBB,
                                          MachineBasicBlock::iterator I,
                                          unsigned DestReg, int FrameIndex,
                                          const TargetRegisterClass *RC,
                                          const TargetRegisterInfo *TRI) const
{
  DebugLoc DL;
  if (I != MBB.end()) DL = I->getDebugLoc();
  BuildMI(MBB, I, DL, get(XCore::LDWFI), DestReg)
    .addFrameIndex(FrameIndex)
    .addImm(0);
}

bool XCoreInstrInfo::spillCalleeSavedRegisters(MachineBasicBlock &MBB,
                                               MachineBasicBlock::iterator MI,
                                        const std::vector<CalleeSavedInfo> &CSI,
                                          const TargetRegisterInfo *TRI) const {
  if (CSI.empty()) {
    return true;
  }
  MachineFunction *MF = MBB.getParent();
  XCoreFunctionInfo *XFI = MF->getInfo<XCoreFunctionInfo>();
  
  bool emitFrameMoves = XCoreRegisterInfo::needsFrameMoves(*MF);

  DebugLoc DL;
  if (MI != MBB.end()) DL = MI->getDebugLoc();
  
  for (std::vector<CalleeSavedInfo>::const_iterator it = CSI.begin();
                                                    it != CSI.end(); ++it) {
    // Add the callee-saved register as live-in. It's killed at the spill.
    MBB.addLiveIn(it->getReg());

    storeRegToStackSlot(MBB, MI, it->getReg(), true,
                        it->getFrameIdx(), it->getRegClass(), &RI);
    if (emitFrameMoves) {
      MCSymbol *SaveLabel = MF->getContext().CreateTempSymbol();
      BuildMI(MBB, MI, DL, get(XCore::DBG_LABEL)).addSym(SaveLabel);
      XFI->getSpillLabels().push_back(std::make_pair(SaveLabel, *it));
    }
  }
  return true;
}

bool XCoreInstrInfo::restoreCalleeSavedRegisters(MachineBasicBlock &MBB,
                                         MachineBasicBlock::iterator MI,
                                        const std::vector<CalleeSavedInfo> &CSI,
                                            const TargetRegisterInfo *TRI) const
{
  bool AtStart = MI == MBB.begin();
  MachineBasicBlock::iterator BeforeI = MI;
  if (!AtStart)
    --BeforeI;
  for (std::vector<CalleeSavedInfo>::const_iterator it = CSI.begin();
                                                    it != CSI.end(); ++it) {
    
    loadRegFromStackSlot(MBB, MI, it->getReg(),
                                  it->getFrameIdx(),
                         it->getRegClass(), &RI);
    assert(MI != MBB.begin() &&
           "loadRegFromStackSlot didn't insert any code!");
    // Insert in reverse order.  loadRegFromStackSlot can insert multiple
    // instructions.
    if (AtStart)
      MI = MBB.begin();
    else {
      MI = BeforeI;
      ++MI;
    }
  }
  return true;
}

/// ReverseBranchCondition - Return the inverse opcode of the 
/// specified Branch instruction.
bool XCoreInstrInfo::
ReverseBranchCondition(SmallVectorImpl<MachineOperand> &Cond) const 
{
  assert((Cond.size() == 2) && 
          "Invalid XCore branch condition!");
  Cond[0].setImm(GetOppositeBranchCondition((XCore::CondCode)Cond[0].getImm()));
  return false;
}
