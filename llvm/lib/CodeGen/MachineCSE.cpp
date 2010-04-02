//===-- MachineCSE.cpp - Machine Common Subexpression Elimination Pass ----===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass performs global common subexpression elimination on machine
// instructions using a scoped hash table based value numbering scheme. It
// must be run while the machine function is still in SSA form.
//
//===----------------------------------------------------------------------===//

#define DEBUG_TYPE "machine-cse"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/Analysis/AliasAnalysis.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/ADT/ScopedHashTable.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

using namespace llvm;

STATISTIC(NumCoalesces, "Number of copies coalesced");
STATISTIC(NumCSEs,      "Number of common subexpression eliminated");

namespace {
  class MachineCSE : public MachineFunctionPass {
    const TargetInstrInfo *TII;
    const TargetRegisterInfo *TRI;
    AliasAnalysis *AA;
    MachineDominatorTree *DT;
    MachineRegisterInfo *MRI;
  public:
    static char ID; // Pass identification
    MachineCSE() : MachineFunctionPass(&ID), CurrVN(0) {}

    virtual bool runOnMachineFunction(MachineFunction &MF);
    
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      MachineFunctionPass::getAnalysisUsage(AU);
      AU.addRequired<AliasAnalysis>();
      AU.addRequired<MachineDominatorTree>();
      AU.addPreserved<MachineDominatorTree>();
    }

  private:
    unsigned CurrVN;
    ScopedHashTable<MachineInstr*, unsigned, MachineInstrExpressionTrait> VNT;
    SmallVector<MachineInstr*, 64> Exps;

    bool PerformTrivialCoalescing(MachineInstr *MI, MachineBasicBlock *MBB);
    bool isPhysDefTriviallyDead(unsigned Reg,
                                MachineBasicBlock::const_iterator I,
                                MachineBasicBlock::const_iterator E);
    bool hasLivePhysRegDefUse(MachineInstr *MI, MachineBasicBlock *MBB);
    bool isCSECandidate(MachineInstr *MI);
    bool isProfitableToCSE(unsigned CSReg, unsigned Reg,
                           MachineInstr *CSMI, MachineInstr *MI);
    bool ProcessBlock(MachineDomTreeNode *Node);
  };
} // end anonymous namespace

char MachineCSE::ID = 0;
static RegisterPass<MachineCSE>
X("machine-cse", "Machine Common Subexpression Elimination");

FunctionPass *llvm::createMachineCSEPass() { return new MachineCSE(); }

bool MachineCSE::PerformTrivialCoalescing(MachineInstr *MI,
                                          MachineBasicBlock *MBB) {
  bool Changed = false;
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg() || !MO.isUse())
      continue;
    unsigned Reg = MO.getReg();
    if (!Reg || TargetRegisterInfo::isPhysicalRegister(Reg))
      continue;
    if (!MRI->hasOneUse(Reg))
      // Only coalesce single use copies. This ensure the copy will be
      // deleted.
      continue;
    MachineInstr *DefMI = MRI->getVRegDef(Reg);
    if (DefMI->getParent() != MBB)
      continue;
    unsigned SrcReg, DstReg, SrcSubIdx, DstSubIdx;
    if (TII->isMoveInstr(*DefMI, SrcReg, DstReg, SrcSubIdx, DstSubIdx) &&
        TargetRegisterInfo::isVirtualRegister(SrcReg) &&
        !SrcSubIdx && !DstSubIdx) {
      const TargetRegisterClass *SRC   = MRI->getRegClass(SrcReg);
      const TargetRegisterClass *RC    = MRI->getRegClass(Reg);
      const TargetRegisterClass *NewRC = getCommonSubClass(RC, SRC);
      if (!NewRC)
        continue;
      DEBUG(dbgs() << "Coalescing: " << *DefMI);
      DEBUG(dbgs() << "*** to: " << *MI);
      MO.setReg(SrcReg);
      if (NewRC != SRC)
        MRI->setRegClass(SrcReg, NewRC);
      DefMI->eraseFromParent();
      ++NumCoalesces;
      Changed = true;
    }
  }

  return Changed;
}

bool MachineCSE::isPhysDefTriviallyDead(unsigned Reg,
                                        MachineBasicBlock::const_iterator I,
                                        MachineBasicBlock::const_iterator E) {
  unsigned LookAheadLeft = 5;
  while (LookAheadLeft) {
    // Skip over dbg_value's.
    while (I != E && I->isDebugValue())
      ++I;

    if (I == E)
      // Reached end of block, register is obviously dead.
      return true;

    bool SeenDef = false;
    for (unsigned i = 0, e = I->getNumOperands(); i != e; ++i) {
      const MachineOperand &MO = I->getOperand(i);
      if (!MO.isReg() || !MO.getReg())
        continue;
      if (!TRI->regsOverlap(MO.getReg(), Reg))
        continue;
      if (MO.isUse())
        return false;
      SeenDef = true;
    }
    if (SeenDef)
      // See a def of Reg (or an alias) before encountering any use, it's 
      // trivially dead.
      return true;

    --LookAheadLeft;
    ++I;
  }
  return false;
}

/// hasLivePhysRegDefUse - Return true if the specified instruction read / write
/// physical registers (except for dead defs of physical registers).
bool MachineCSE::hasLivePhysRegDefUse(MachineInstr *MI, MachineBasicBlock *MBB){
  unsigned PhysDef = 0;
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg())
      continue;
    unsigned Reg = MO.getReg();
    if (!Reg)
      continue;
    if (TargetRegisterInfo::isPhysicalRegister(Reg)) {
      if (MO.isUse())
        // Can't touch anything to read a physical register.
        return true;
      if (MO.isDead())
        // If the def is dead, it's ok.
        continue;
      // Ok, this is a physical register def that's not marked "dead". That's
      // common since this pass is run before livevariables. We can scan
      // forward a few instructions and check if it is obviously dead.
      if (PhysDef)
        // Multiple physical register defs. These are rare, forget about it.
        return true;
      PhysDef = Reg;
    }
  }

  if (PhysDef) {
    MachineBasicBlock::iterator I = MI; I = llvm::next(I);
    if (!isPhysDefTriviallyDead(PhysDef, I, MBB->end()))
      return true;
  }
  return false;
}

static bool isCopy(const MachineInstr *MI, const TargetInstrInfo *TII) {
  unsigned SrcReg, DstReg, SrcSubIdx, DstSubIdx;
  return TII->isMoveInstr(*MI, SrcReg, DstReg, SrcSubIdx, DstSubIdx) ||
    MI->isExtractSubreg() || MI->isInsertSubreg() || MI->isSubregToReg();
}

bool MachineCSE::isCSECandidate(MachineInstr *MI) {
  if (MI->isLabel() || MI->isPHI() || MI->isImplicitDef() ||
      MI->isKill() || MI->isInlineAsm() || MI->isDebugValue())
    return false;

  // Ignore copies.
  if (isCopy(MI, TII))
    return false;

  // Ignore stuff that we obviously can't move.
  const TargetInstrDesc &TID = MI->getDesc();  
  if (TID.mayStore() || TID.isCall() || TID.isTerminator() ||
      TID.hasUnmodeledSideEffects())
    return false;

  if (TID.mayLoad()) {
    // Okay, this instruction does a load. As a refinement, we allow the target
    // to decide whether the loaded value is actually a constant. If so, we can
    // actually use it as a load.
    if (!MI->isInvariantLoad(AA))
      // FIXME: we should be able to hoist loads with no other side effects if
      // there are no other instructions which can change memory in this loop.
      // This is a trivial form of alias analysis.
      return false;
  }
  return true;
}

/// isProfitableToCSE - Return true if it's profitable to eliminate MI with a
/// common expression that defines Reg.
bool MachineCSE::isProfitableToCSE(unsigned CSReg, unsigned Reg,
                                   MachineInstr *CSMI, MachineInstr *MI) {
  // FIXME: Heuristics that works around the lack the live range splitting.

  // Heuristics #1: Don't cse "cheap" computating if the def is not local or in an
  // immediate predecessor. We don't want to increase register pressure and end up
  // causing other computation to be spilled.
  if (MI->getDesc().isAsCheapAsAMove()) {
    MachineBasicBlock *CSBB = CSMI->getParent();
    MachineBasicBlock *BB = MI->getParent();
    if (CSBB != BB && 
        find(CSBB->succ_begin(), CSBB->succ_end(), BB) == CSBB->succ_end())
      return false;
  }

  // Heuristics #2: If the expression doesn't not use a vr and the only use
  // of the redundant computation are copies, do not cse.
  bool HasVRegUse = false;
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    const MachineOperand &MO = MI->getOperand(i);
    if (MO.isReg() && MO.isUse() && MO.getReg() &&
        TargetRegisterInfo::isVirtualRegister(MO.getReg())) {
      HasVRegUse = true;
      break;
    }
  }
  if (!HasVRegUse) {
    bool HasNonCopyUse = false;
    for (MachineRegisterInfo::use_nodbg_iterator I =  MRI->use_nodbg_begin(Reg),
           E = MRI->use_nodbg_end(); I != E; ++I) {
      MachineInstr *Use = &*I;
      // Ignore copies.
      if (!isCopy(Use, TII)) {
        HasNonCopyUse = true;
        break;
      }
    }
    if (!HasNonCopyUse)
      return false;
  }

  // Heuristics #3: If the common subexpression is used by PHIs, do not reuse
  // it unless the defined value is already used in the BB of the new use.
  bool HasPHI = false;
  SmallPtrSet<MachineBasicBlock*, 4> CSBBs;
  for (MachineRegisterInfo::use_nodbg_iterator I =  MRI->use_nodbg_begin(CSReg),
       E = MRI->use_nodbg_end(); I != E; ++I) {
    MachineInstr *Use = &*I;
    HasPHI |= Use->isPHI();
    CSBBs.insert(Use->getParent());
  }

  if (!HasPHI)
    return true;
  return CSBBs.count(MI->getParent());
}

bool MachineCSE::ProcessBlock(MachineDomTreeNode *Node) {
  bool Changed = false;

  SmallVector<std::pair<unsigned, unsigned>, 8> CSEPairs;
  ScopedHashTableScope<MachineInstr*, unsigned,
    MachineInstrExpressionTrait> VNTS(VNT);
  MachineBasicBlock *MBB = Node->getBlock();
  for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end(); I != E; ) {
    MachineInstr *MI = &*I;
    ++I;

    if (!isCSECandidate(MI))
      continue;

    bool FoundCSE = VNT.count(MI);
    if (!FoundCSE) {
      // Look for trivial copy coalescing opportunities.
      if (PerformTrivialCoalescing(MI, MBB)) {
        // After coalescing MI itself may become a copy.
        if (isCopy(MI, TII))
          continue;
        FoundCSE = VNT.count(MI);
      }
    }
    // FIXME: commute commutable instructions?

    // If the instruction defines a physical register and the value *may* be
    // used, then it's not safe to replace it with a common subexpression.
    if (FoundCSE && hasLivePhysRegDefUse(MI, MBB))
      FoundCSE = false;

    if (!FoundCSE) {
      VNT.insert(MI, CurrVN++);
      Exps.push_back(MI);
      continue;
    }

    // Found a common subexpression, eliminate it.
    unsigned CSVN = VNT.lookup(MI);
    MachineInstr *CSMI = Exps[CSVN];
    DEBUG(dbgs() << "Examining: " << *MI);
    DEBUG(dbgs() << "*** Found a common subexpression: " << *CSMI);

    // Check if it's profitable to perform this CSE.
    bool DoCSE = true;
    unsigned NumDefs = MI->getDesc().getNumDefs();
    for (unsigned i = 0, e = MI->getNumOperands(); NumDefs && i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (!MO.isReg() || !MO.isDef())
        continue;
      unsigned OldReg = MO.getReg();
      unsigned NewReg = CSMI->getOperand(i).getReg();
      if (OldReg == NewReg)
        continue;
      assert(TargetRegisterInfo::isVirtualRegister(OldReg) &&
             TargetRegisterInfo::isVirtualRegister(NewReg) &&
             "Do not CSE physical register defs!");
      if (!isProfitableToCSE(NewReg, OldReg, CSMI, MI)) {
        DoCSE = false;
        break;
      }
      CSEPairs.push_back(std::make_pair(OldReg, NewReg));
      --NumDefs;
    }

    // Actually perform the elimination.
    if (DoCSE) {
      for (unsigned i = 0, e = CSEPairs.size(); i != e; ++i)
        MRI->replaceRegWith(CSEPairs[i].first, CSEPairs[i].second);
      MI->eraseFromParent();
      ++NumCSEs;
    } else {
      DEBUG(dbgs() << "*** Not profitable, avoid CSE!\n");
      VNT.insert(MI, CurrVN++);
      Exps.push_back(MI);
    }
    CSEPairs.clear();
  }

  // Recursively call ProcessBlock with childred.
  const std::vector<MachineDomTreeNode*> &Children = Node->getChildren();
  for (unsigned i = 0, e = Children.size(); i != e; ++i)
    Changed |= ProcessBlock(Children[i]);

  return Changed;
}

bool MachineCSE::runOnMachineFunction(MachineFunction &MF) {
  TII = MF.getTarget().getInstrInfo();
  TRI = MF.getTarget().getRegisterInfo();
  MRI = &MF.getRegInfo();
  AA = &getAnalysis<AliasAnalysis>();
  DT = &getAnalysis<MachineDominatorTree>();
  return ProcessBlock(DT->getRootNode());
}
