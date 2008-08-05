//===-- UnreachableBlockElim.cpp - Remove unreachable blocks for codegen --===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
// This pass is an extremely simple version of the SimplifyCFG pass.  Its sole
// job is to delete LLVM basic blocks that are not reachable from the entry
// node.  To do this, it performs a simple depth first traversal of the CFG,
// then deletes any unvisited nodes.
//
// Note that this pass is really a hack.  In particular, the instruction
// selectors for various targets should just not generate code for unreachable
// blocks.  Until LLVM has a more systematic way of defining instruction
// selectors, however, we cannot really expect them to handle additional
// complexity.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/Passes.h"
#include "llvm/Constant.h"
#include "llvm/Instructions.h"
#include "llvm/Function.h"
#include "llvm/Pass.h"
#include "llvm/Type.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/Support/CFG.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/ADT/DepthFirstIterator.h"
using namespace llvm;

namespace {
  class VISIBILITY_HIDDEN UnreachableBlockElim : public FunctionPass {
    virtual bool runOnFunction(Function &F);
  public:
    static char ID; // Pass identification, replacement for typeid
    UnreachableBlockElim() : FunctionPass((intptr_t)&ID) {}
  };
}
char UnreachableBlockElim::ID = 0;
static RegisterPass<UnreachableBlockElim>
X("unreachableblockelim", "Remove unreachable blocks from the CFG");

FunctionPass *llvm::createUnreachableBlockEliminationPass() {
  return new UnreachableBlockElim();
}

bool UnreachableBlockElim::runOnFunction(Function &F) {
  std::set<BasicBlock*> Reachable;

  // Mark all reachable blocks.
  for (df_ext_iterator<Function*> I = df_ext_begin(&F, Reachable),
         E = df_ext_end(&F, Reachable); I != E; ++I)
    /* Mark all reachable blocks */;

  // Loop over all dead blocks, remembering them and deleting all instructions
  // in them.
  std::vector<BasicBlock*> DeadBlocks;
  for (Function::iterator I = F.begin(), E = F.end(); I != E; ++I)
    if (!Reachable.count(I)) {
      BasicBlock *BB = I;
      DeadBlocks.push_back(BB);
      while (PHINode *PN = dyn_cast<PHINode>(BB->begin())) {
        PN->replaceAllUsesWith(Constant::getNullValue(PN->getType()));
        BB->getInstList().pop_front();
      }
      for (succ_iterator SI = succ_begin(BB), E = succ_end(BB); SI != E; ++SI)
        (*SI)->removePredecessor(BB);
      BB->dropAllReferences();
    }

  // Actually remove the blocks now.
  for (unsigned i = 0, e = DeadBlocks.size(); i != e; ++i)
    DeadBlocks[i]->eraseFromParent();

  return DeadBlocks.size();
}


namespace {
  class VISIBILITY_HIDDEN UnreachableMachineBlockElim : 
        public MachineFunctionPass {
    virtual bool runOnMachineFunction(MachineFunction &F);
    
  public:
    static char ID; // Pass identification, replacement for typeid
    UnreachableMachineBlockElim() : MachineFunctionPass((intptr_t)&ID) {}
  };
}
char UnreachableMachineBlockElim::ID = 0;

static RegisterPass<UnreachableMachineBlockElim>
Y("unreachable-mbb-elimination",
  "Remove unreachable machine basic blocks");

const PassInfo *const llvm::UnreachableMachineBlockElimID = &Y;

bool UnreachableMachineBlockElim::runOnMachineFunction(MachineFunction &F) {
  std::set<MachineBasicBlock*> Reachable;

  // Mark all reachable blocks.
  for (df_ext_iterator<MachineFunction*> I = df_ext_begin(&F, Reachable),
         E = df_ext_end(&F, Reachable); I != E; ++I)
    /* Mark all reachable blocks */;

  // Loop over all dead blocks, remembering them and deleting all instructions
  // in them.
  std::vector<MachineBasicBlock*> DeadBlocks;
  for (MachineFunction::iterator I = F.begin(), E = F.end(); I != E; ++I)
    if (!Reachable.count(I)) {
      MachineBasicBlock *BB = I;
      DeadBlocks.push_back(BB);
      
      while (BB->succ_begin() != BB->succ_end()) {
        MachineBasicBlock* succ = *BB->succ_begin();
        
        MachineBasicBlock::iterator start = succ->begin();
        while (start != succ->end() &&
               start->getOpcode() == TargetInstrInfo::PHI) {
          for (unsigned i = start->getNumOperands() - 1; i >= 2; i-=2)
            if (start->getOperand(i).isMBB() &&
                start->getOperand(i).getMBB() == BB) {
              start->RemoveOperand(i);
              start->RemoveOperand(i-1);
            }
          
          if (start->getNumOperands() == 1) {
            MachineInstr* phi = start;
            start++;
            phi->eraseFromParent();
          } else
            start++;
        }
        
        BB->removeSuccessor(BB->succ_begin());
      }
    }

  // Actually remove the blocks now.
  for (unsigned i = 0, e = DeadBlocks.size(); i != e; ++i)
    DeadBlocks[i]->eraseFromParent();

  F.RenumberBlocks();

  return DeadBlocks.size();
}

