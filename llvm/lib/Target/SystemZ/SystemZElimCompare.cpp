//===-- SystemZElimCompare.cpp - Eliminate comparison instructions --------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass:
// (1) tries to remove compares if CC already contains the required information
// (2) fuses compares and branches into COMPARE AND BRANCH instructions
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "systemz-elim-compare"

#include "SystemZTargetMachine.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstrBuilder.h"
#include "llvm/IR/Function.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/Target/TargetRegisterInfo.h"

using namespace llvm;

STATISTIC(EliminatedComparisons, "Number of eliminated comparisons");
STATISTIC(FusedComparisons, "Number of fused compare-and-branch instructions");

namespace {
  class SystemZElimCompare : public MachineFunctionPass {
  public:
    static char ID;
    SystemZElimCompare(const SystemZTargetMachine &tm)
      : MachineFunctionPass(ID), TII(0), TRI(0) {}

    virtual const char *getPassName() const {
      return "SystemZ Comparison Elimination";
    }

    bool processBlock(MachineBasicBlock *MBB);
    bool runOnMachineFunction(MachineFunction &F);

  private:
    bool adjustCCMasksForInstr(MachineInstr *MI, MachineInstr *Compare,
                               SmallVectorImpl<MachineInstr *> &CCUsers);
    bool optimizeCompareZero(MachineInstr *Compare,
                             SmallVectorImpl<MachineInstr *> &CCUsers);
    bool fuseCompareAndBranch(MachineInstr *Compare,
                              SmallVectorImpl<MachineInstr *> &CCUsers);

    const SystemZInstrInfo *TII;
    const TargetRegisterInfo *TRI;
  };

  char SystemZElimCompare::ID = 0;
} // end of anonymous namespace

FunctionPass *llvm::createSystemZElimComparePass(SystemZTargetMachine &TM) {
  return new SystemZElimCompare(TM);
}

// Return true if CC is live out of MBB.
static bool isCCLiveOut(MachineBasicBlock *MBB) {
  for (MachineBasicBlock::succ_iterator SI = MBB->succ_begin(),
         SE = MBB->succ_end(); SI != SE; ++SI)
    if ((*SI)->isLiveIn(SystemZ::CC))
      return true;
  return false;
}

// Return true if any CC result of MI would reflect the value of subreg
// SubReg of Reg.
static bool resultTests(MachineInstr *MI, unsigned Reg, unsigned SubReg) {
  if (MI->getNumOperands() > 0 &&
      MI->getOperand(0).isReg() &&
      MI->getOperand(0).isDef() &&
      MI->getOperand(0).getReg() == Reg &&
      MI->getOperand(0).getSubReg() == SubReg)
    return true;

  return false;
}

// The CC users in CCUsers are testing the result of a comparison of some
// value X against zero and we know that any CC value produced by MI
// would also reflect the value of X.  Try to adjust CCUsers so that
// they test the result of MI directly, returning true on success.
// Leave everything unchanged on failure.
bool SystemZElimCompare::
adjustCCMasksForInstr(MachineInstr *MI, MachineInstr *Compare,
                      SmallVectorImpl<MachineInstr *> &CCUsers) {
  int Opcode = MI->getOpcode();
  const MCInstrDesc &Desc = TII->get(Opcode);
  unsigned MIFlags = Desc.TSFlags;

  // See which compare-style condition codes are available.
  unsigned ReusableCCMask = 0;
  if (MIFlags & SystemZII::CCHasZero)
    ReusableCCMask |= SystemZ::CCMASK_CMP_EQ;

  // For unsigned comparisons with zero, only equality makes sense.
  unsigned CompareFlags = Compare->getDesc().TSFlags;
  if (!(CompareFlags & SystemZII::IsLogical) &&
      (MIFlags & SystemZII::CCHasOrder))
    ReusableCCMask |= SystemZ::CCMASK_CMP_LT | SystemZ::CCMASK_CMP_GT;

  if (ReusableCCMask == 0)
    return false;

  unsigned CCValues = SystemZII::getCCValues(MIFlags);
  assert((ReusableCCMask & ~CCValues) == 0 && "Invalid CCValues");

  // Now check whether these flags are enough for all users.
  SmallVector<MachineOperand *, 4> AlterMasks;
  for (unsigned int I = 0, E = CCUsers.size(); I != E; ++I) {
    MachineInstr *MI = CCUsers[I];

    // Fail if this isn't a use of CC that we understand.
    unsigned Flags = MI->getDesc().TSFlags;
    unsigned FirstOpNum;
    if (Flags & SystemZII::CCMaskFirst)
      FirstOpNum = 0;
    else if (Flags & SystemZII::CCMaskLast)
      FirstOpNum = MI->getNumExplicitOperands() - 2;
    else
      return false;

    // Check whether the instruction predicate treats all CC values
    // outside of ReusableCCMask in the same way.  In that case it
    // doesn't matter what those CC values mean.
    unsigned CCValid = MI->getOperand(FirstOpNum).getImm();
    unsigned CCMask = MI->getOperand(FirstOpNum + 1).getImm();
    unsigned OutValid = ~ReusableCCMask & CCValid;
    unsigned OutMask = ~ReusableCCMask & CCMask;
    if (OutMask != 0 && OutMask != OutValid)
      return false;

    AlterMasks.push_back(&MI->getOperand(FirstOpNum));
    AlterMasks.push_back(&MI->getOperand(FirstOpNum + 1));
  }

  // All users are OK.  Adjust the masks for MI.
  for (unsigned I = 0, E = AlterMasks.size(); I != E; I += 2) {
    AlterMasks[I]->setImm(CCValues);
    unsigned CCMask = AlterMasks[I + 1]->getImm();
    if (CCMask & ~ReusableCCMask)
      AlterMasks[I + 1]->setImm((CCMask & ReusableCCMask) |
                                (CCValues & ~ReusableCCMask));
  }

  // CC is now live after MI.
  int CCDef = MI->findRegisterDefOperandIdx(SystemZ::CC, false, true, TRI);
  assert(CCDef >= 0 && "Couldn't find CC set");
  MI->getOperand(CCDef).setIsDead(false);

  // Clear any intervening kills of CC.
  MachineBasicBlock::iterator MBBI = MI, MBBE = Compare;
  for (++MBBI; MBBI != MBBE; ++MBBI)
    MBBI->clearRegisterKills(SystemZ::CC, TRI);

  return true;
}

// Try to optimize cases where comparison instruction Compare is testing
// a value against zero.  Return true on success and if Compare should be
// deleted as dead.  CCUsers is the list of instructions that use the CC
// value produced by Compare.
bool SystemZElimCompare::
optimizeCompareZero(MachineInstr *Compare,
                    SmallVectorImpl<MachineInstr *> &CCUsers) {
  // Check whether this is a comparison against zero.
  if (Compare->getNumExplicitOperands() != 2 ||
      !Compare->getOperand(1).isImm() ||
      Compare->getOperand(1).getImm() != 0)
    return false;

  // Search back for CC results that are based on the first operand.
  unsigned SrcReg = Compare->getOperand(0).getReg();
  unsigned SrcSubReg = Compare->getOperand(0).getSubReg();
  MachineBasicBlock *MBB = Compare->getParent();
  MachineBasicBlock::iterator MBBI = Compare, MBBE = MBB->begin();
  while (MBBI != MBBE) {
    --MBBI;
    MachineInstr *MI = MBBI;
    if (resultTests(MI, SrcReg, SrcSubReg) &&
        adjustCCMasksForInstr(MI, Compare, CCUsers)) {
      EliminatedComparisons += 1;
      return true;
    }
    if (MI->modifiesRegister(SrcReg, TRI) ||
        MI->modifiesRegister(SystemZ::CC, TRI))
      return false;
  }
  return false;
}

// Try to fuse comparison instruction Compare into a later branch.
// Return true on success and if Compare is therefore redundant.
bool SystemZElimCompare::
fuseCompareAndBranch(MachineInstr *Compare,
                     SmallVectorImpl<MachineInstr *> &CCUsers) {
  // See whether we have a comparison that can be fused.
  unsigned FusedOpcode = TII->getCompareAndBranch(Compare->getOpcode(),
                                                  Compare);
  if (!FusedOpcode)
    return false;

  // See whether we have a single branch with which to fuse.
  if (CCUsers.size() != 1)
    return false;
  MachineInstr *Branch = CCUsers[0];
  if (Branch->getOpcode() != SystemZ::BRC)
    return false;

  // Make sure that the operands are available at the branch.
  unsigned SrcReg = Compare->getOperand(0).getReg();
  unsigned SrcReg2 = (Compare->getOperand(1).isReg() ?
                      Compare->getOperand(1).getReg() : 0);
  MachineBasicBlock::iterator MBBI = Compare, MBBE = Branch;
  for (++MBBI; MBBI != MBBE; ++MBBI)
    if (MBBI->modifiesRegister(SrcReg, TRI) ||
        (SrcReg2 && MBBI->modifiesRegister(SrcReg2, TRI)))
      return false;

  // Read the branch mask and target.
  MachineOperand CCMask(MBBI->getOperand(1));
  MachineOperand Target(MBBI->getOperand(2));
  assert((CCMask.getImm() & ~SystemZ::CCMASK_ICMP) == 0 &&
         "Invalid condition-code mask for integer comparison");

  // Clear out all current operands.
  int CCUse = MBBI->findRegisterUseOperandIdx(SystemZ::CC, false, TRI);
  assert(CCUse >= 0 && "BRC must use CC");
  Branch->RemoveOperand(CCUse);
  Branch->RemoveOperand(2);
  Branch->RemoveOperand(1);
  Branch->RemoveOperand(0);

  // Rebuild Branch as a fused compare and branch.
  Branch->setDesc(TII->get(FusedOpcode));
  MachineInstrBuilder(*Branch->getParent()->getParent(), Branch)
    .addOperand(Compare->getOperand(0))
    .addOperand(Compare->getOperand(1))
    .addOperand(CCMask)
    .addOperand(Target)
    .addReg(SystemZ::CC, RegState::ImplicitDefine);

  // Clear any intervening kills of SrcReg and SrcReg2.
  MBBI = Compare;
  for (++MBBI; MBBI != MBBE; ++MBBI) {
    MBBI->clearRegisterKills(SrcReg, TRI);
    if (SrcReg2)
      MBBI->clearRegisterKills(SrcReg2, TRI);
  }
  FusedComparisons += 1;
  return true;
}

// Process all comparison instructions in MBB.  Return true if something
// changed.
bool SystemZElimCompare::processBlock(MachineBasicBlock *MBB) {
  bool Changed = false;

  // Walk backwards through the block looking for comparisons, recording
  // all CC users as we go.  The subroutines can delete Compare and
  // instructions before it.
  bool CompleteCCUsers = !isCCLiveOut(MBB);
  SmallVector<MachineInstr *, 4> CCUsers;
  MachineBasicBlock::iterator MBBI = MBB->end();
  while (MBBI != MBB->begin()) {
    MachineInstr *MI = --MBBI;
    if (CompleteCCUsers &&
        MI->isCompare() &&
        (optimizeCompareZero(MI, CCUsers) ||
         fuseCompareAndBranch(MI, CCUsers))) {
      ++MBBI;
      MI->removeFromParent();
      Changed = true;
      CCUsers.clear();
      CompleteCCUsers = true;
      continue;
    }

    if (MI->definesRegister(SystemZ::CC, TRI)) {
      CCUsers.clear();
      CompleteCCUsers = true;
    } else if (MI->modifiesRegister(SystemZ::CC, TRI))
      CompleteCCUsers = false;

    if (CompleteCCUsers && MI->readsRegister(SystemZ::CC, TRI))
      CCUsers.push_back(MI);
  }
  return Changed;
}

bool SystemZElimCompare::runOnMachineFunction(MachineFunction &F) {
  TII = static_cast<const SystemZInstrInfo *>(F.getTarget().getInstrInfo());
  TRI = &TII->getRegisterInfo();

  bool Changed = false;
  for (MachineFunction::iterator MFI = F.begin(), MFE = F.end();
       MFI != MFE; ++MFI)
    Changed |= processBlock(MFI);

  return Changed;
}
