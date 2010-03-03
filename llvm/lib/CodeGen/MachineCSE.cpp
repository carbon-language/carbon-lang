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

namespace llvm {
  template<> struct DenseMapInfo<MachineInstr*> {
    static inline MachineInstr *getEmptyKey() {
      return 0;
    }

    static inline MachineInstr *getTombstoneKey() {
      return reinterpret_cast<MachineInstr*>(-1);
    }

    static unsigned getHashValue(const MachineInstr* const &MI) {
      unsigned Hash = MI->getOpcode() * 37;
      for (unsigned i = 0, e = MI->getNumOperands(); i != e; ++i) {
        const MachineOperand &MO = MI->getOperand(i);
        uint64_t Key = (uint64_t)MO.getType() << 32;
        switch (MO.getType()) {
        default: break;
        case MachineOperand::MO_Register:
          if (MO.isDef() && TargetRegisterInfo::isVirtualRegister(MO.getReg()))
            continue;  // Skip virtual register defs.
          Key |= MO.getReg();
          break;
        case MachineOperand::MO_Immediate:
          Key |= MO.getImm();
          break;
        case MachineOperand::MO_FrameIndex:
        case MachineOperand::MO_ConstantPoolIndex:
        case MachineOperand::MO_JumpTableIndex:
          Key |= MO.getIndex();
          break;
        case MachineOperand::MO_MachineBasicBlock:
          Key |= DenseMapInfo<void*>::getHashValue(MO.getMBB());
          break;
        case MachineOperand::MO_GlobalAddress:
          Key |= DenseMapInfo<void*>::getHashValue(MO.getGlobal());
          break;
        case MachineOperand::MO_BlockAddress:
          Key |= DenseMapInfo<void*>::getHashValue(MO.getBlockAddress());
          break;
        }
        Key += ~(Key << 32);
        Key ^= (Key >> 22);
        Key += ~(Key << 13);
        Key ^= (Key >> 8);
        Key += (Key << 3);
        Key ^= (Key >> 15);
        Key += ~(Key << 27);
        Key ^= (Key >> 31);
        Hash = (unsigned)Key + Hash * 37;
      }
      return Hash;
    }

    static bool isEqual(const MachineInstr* const &LHS,
                        const MachineInstr* const &RHS) {
      if (RHS == getEmptyKey() || RHS == getTombstoneKey() ||
          LHS == getEmptyKey() || LHS == getTombstoneKey())
        return LHS == RHS;
      return LHS->isIdenticalTo(RHS, MachineInstr::IgnoreVRegDefs);
    }
  };
} // end llvm namespace

namespace {
  class MachineCSE : public MachineFunctionPass {
    const TargetInstrInfo *TII;
    MachineRegisterInfo  *MRI;
    MachineDominatorTree *DT;
    ScopedHashTable<MachineInstr*, unsigned> VNT;
    unsigned CurrVN;
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
    if (MO.isReg() && MO.isUse()) {
      unsigned Reg = MO.getReg();
      if (!Reg || TargetRegisterInfo::isPhysicalRegister(Reg))
        continue;
      MachineInstr *DefMI = MRI->getVRegDef(Reg);
      if (DefMI->getParent() == MBB) {
        unsigned SrcReg, DstReg, SrcSubIdx, DstSubIdx;
        if (TII->isMoveInstr(*DefMI, SrcReg, DstReg, SrcSubIdx, DstSubIdx) &&
            TargetRegisterInfo::isVirtualRegister(SrcReg) &&
            !SrcSubIdx && !DstSubIdx) {
          MO.setReg(SrcReg);
          Changed = true;
        }
      }
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
    if (TargetRegisterInfo::isPhysicalRegister(Reg) &&
        !(MO.isDef() && MO.isDead()))
      return true;
  }
  return false;
}

bool MachineCSE::ProcessBlock(MachineDomTreeNode *Node) {
  bool Changed = false;

  ScopedHashTableScope<MachineInstr*, unsigned> VNTS(VNT);
  MachineBasicBlock *MBB = Node->getBlock();
  for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end(); I != E;
       ++I) {
    MachineInstr *MI = &*I;
    bool SawStore = false;
    if (!MI->isSafeToMove(TII, 0, SawStore))
      continue;
    // Ignore copies or instructions that read / write physical registers
    // (except for dead defs of physical registers).
    unsigned SrcReg, DstReg, SrcSubIdx, DstSubIdx;
    if (TII->isMoveInstr(*MI, SrcReg, DstReg, SrcSubIdx, DstSubIdx))
      continue;
    if (hasLivePhysRegDefUse(MI))
      continue;

    bool FoundCSE = VNT.count(MI);
    if (!FoundCSE) {
      // Look for trivial copy coalescing opportunities.
      if (PerformTrivialCoalescing(MI, MBB))
        FoundCSE = VNT.count(MI);
    }

    if (FoundCSE)
      DEBUG(dbgs() << "Found a common subexpression: " << *MI);
    else
      VNT.insert(MI, ++CurrVN);
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
