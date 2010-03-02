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
      return LHS->isIdenticalTo(RHS);
    }
  };
} // end llvm namespace

namespace {
  class MachineCSE : public MachineFunctionPass {
    ScopedHashTable<MachineInstr*, unsigned> VNT;
    MachineDominatorTree *DT;
  public:
    static char ID; // Pass identification
    MachineCSE() : MachineFunctionPass(&ID) {}

    virtual bool runOnMachineFunction(MachineFunction &MF);
    
    virtual void getAnalysisUsage(AnalysisUsage &AU) const {
      AU.setPreservesCFG();
      MachineFunctionPass::getAnalysisUsage(AU);
      AU.addRequired<MachineDominatorTree>();
      AU.addPreserved<MachineDominatorTree>();
    }

  private:
    bool ProcessBlock(MachineDomTreeNode *Node);
  };
} // end anonymous namespace

char MachineCSE::ID = 0;
static RegisterPass<MachineCSE>
X("machine-cse", "Machine Common Subexpression Elimination");

FunctionPass *llvm::createMachineCSEPass() { return new MachineCSE(); }

bool MachineCSE::ProcessBlock(MachineDomTreeNode *Node) {
  ScopedHashTableScope<MachineInstr*, unsigned> VNTS(VNT);
  MachineBasicBlock *MBB = Node->getBlock();
  for (MachineBasicBlock::iterator I = MBB->begin(), E = MBB->end(); I != E;
       ++I) {
  }
  return false;
}

bool MachineCSE::runOnMachineFunction(MachineFunction &MF) {
  DT = &getAnalysis<MachineDominatorTree>();
  return ProcessBlock(DT->getRootNode());
}
