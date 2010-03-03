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
    MachineRegisterInfo  *MRI;
    MachineDominatorTree *DT;
  public:
    static char ID; // Pass identification
    MachineCSE() : MachineFunctionPass(&ID), CurrVN(0) {}

    virtual bool runOnMachineFunction(MachineFunction &MF);
    
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      MachineFunctionPass::getAnalysisUsage(AU);
      AU.addRequired<MachineDominatorTree>();
      AU.addPreserved<MachineDominatorTree>();
    }

  private:
    unsigned CurrVN;
    ScopedHashTable<MachineInstr*, unsigned, MachineInstrExpressionTrait> VNT;
    SmallVector<MachineInstr*, 64> Exps;

    bool PerformTrivialCoalescing(MachineInstr *MI, MachineBasicBlock *MBB);
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
      MO.setReg(SrcReg);
      DefMI->eraseFromParent();
      ++NumCoalesces;
      Changed = true;
    }
  }

  return Changed;
}

static bool hasLivePhysRegDefUse(MachineInstr *MI) {
  for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
    MachineOperand &MO = MI->getOperand(i);
    if (!MO.isReg())
      continue;
    unsigned Reg = MO.getReg();
    if (!Reg)
      continue;
    // FIXME: This is obviously overly conservative. On x86 lots of instructions
    // will def EFLAGS and they are not marked dead at this point.
    if (TargetRegisterInfo::isPhysicalRegister(Reg) &&
        !(MO.isDef() && MO.isDead()))
      return true;
  }
  return false;
}

bool MachineCSE::ProcessBlock(MachineDomTreeNode *Node) {
  bool Changed = false;

  ScopedHashTableScope<MachineInstr*, unsigned,
    MachineInstrExpressionTrait> VNTS(VNT);
  MachineBasicBlock *MBB = Node->getBlock();
  for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end(); I != E; ) {
    MachineInstr *MI = &*I;
    ++I;
    bool SawStore = false;
    if (!MI->isSafeToMove(TII, 0, SawStore))
      continue;
    // Ignore copies or instructions that read / write physical registers
    // (except for dead defs of physical registers).
    unsigned SrcReg, DstReg, SrcSubIdx, DstSubIdx;
    if (TII->isMoveInstr(*MI, SrcReg, DstReg, SrcSubIdx, DstSubIdx) ||
        MI->isExtractSubreg() || MI->isInsertSubreg() || MI->isSubregToReg())
      continue;    

    bool FoundCSE = VNT.count(MI);
    if (!FoundCSE) {
      // Look for trivial copy coalescing opportunities.
      if (PerformTrivialCoalescing(MI, MBB))
        FoundCSE = VNT.count(MI);
    }

    // If the instruction defines a physical register and the value *may* be
    // used, then it's not safe to replace it with a common subexpression.
    if (FoundCSE && hasLivePhysRegDefUse(MI))
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
    unsigned NumDefs = MI->getDesc().getNumDefs();
    for (unsigned i = 0, e = MI->getNumOperands(); NumDefs && i != e; ++i) {
      MachineOperand &MO = MI->getOperand(i);
      if (!MO.isReg() || !MO.isDef())
        continue;
      unsigned OldReg = MO.getReg();
      unsigned NewReg = CSMI->getOperand(i).getReg();
      assert(OldReg != NewReg &&
             TargetRegisterInfo::isVirtualRegister(OldReg) &&
             TargetRegisterInfo::isVirtualRegister(NewReg) &&
             "Do not CSE physical register defs!");
      MRI->replaceRegWith(OldReg, NewReg);
      --NumDefs;
    }
    MI->eraseFromParent();
    ++NumCSEs;
  }

  // Recursively call ProcessBlock with childred.
  const std::vector<MachineDomTreeNode*> &Children = Node->getChildren();
  for (unsigned i = 0, e = Children.size(); i != e; ++i)
    Changed |= ProcessBlock(Children[i]);

  return Changed;
}

bool MachineCSE::runOnMachineFunction(MachineFunction &MF) {
  TII = MF.getTarget().getInstrInfo();
  MRI = &MF.getRegInfo();
  DT = &getAnalysis<MachineDominatorTree>();
  return ProcessBlock(DT->getRootNode());
}
