//===----------- BreakCriticalMachineEdges - Break critical edges ---------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file was developed by Fernando Pereira and is distributed under
// the University of Illinois Open Source License. See LICENSE.TXT for details.
//
//===---------------------------------------------------------------------===//
//
// Break all of the critical edges in the CFG by inserting a dummy basic block. 
// This pass may be "required" by passes that cannot deal with critical edges.
// Notice that this pass invalidates the CFG, because the same BasicBlock is 
// used as parameter for the src MachineBasicBlock and the new dummy
// MachineBasicBlock.
//
//===---------------------------------------------------------------------===//

#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineInstr.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/Target/TargetInstrInfo.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Compiler.h"

using namespace llvm;

namespace {
  struct VISIBILITY_HIDDEN BreakCriticalMachineEdges :
                           public MachineFunctionPass {
    static char ID; // Pass identification
    BreakCriticalMachineEdges() : MachineFunctionPass((intptr_t)&ID) {}
    
    bool runOnMachineFunction(MachineFunction& Fn);
    void splitCriticalEdge(MachineBasicBlock* A, MachineBasicBlock* B);
  };
  
  char BreakCriticalMachineEdges::ID = 0;
  RegisterPass<BreakCriticalMachineEdges> X("critical-machine-edges",
                                            "Break critical machine code edges");
}

const PassInfo *llvm::BreakCriticalMachineEdgesID = X.getPassInfo();

void BreakCriticalMachineEdges::splitCriticalEdge(MachineBasicBlock* src,
                                                  MachineBasicBlock* dst) {
  const BasicBlock* srcBB = src->getBasicBlock();

  MachineBasicBlock* crit_mbb = new MachineBasicBlock(srcBB);

  // modify the llvm control flow graph
  src->removeSuccessor(dst);
  src->addSuccessor(crit_mbb);
  crit_mbb->addSuccessor(dst);

  // insert the new block into the machine function.
  src->getParent()->getBasicBlockList().insert(src->getParent()->end(),
                                               crit_mbb);

  // insert a unconditional branch linking the new block to dst
  const TargetMachine& TM = src->getParent()->getTarget();
  const TargetInstrInfo* TII = TM.getInstrInfo();
  std::vector<MachineOperand> emptyConditions;
  TII->InsertBranch(*crit_mbb, dst, (MachineBasicBlock*)0, emptyConditions);

  // modify every branch in src that points to dst to point to the new
  // machine basic block instead:
  MachineBasicBlock::iterator mii = src->end();
  bool found_branch = false;
  while (mii != src->begin()) {
    mii--;
    // if there are no more branches, finish the loop
    if (!TII->isTerminatorInstr(mii->getOpcode())) {
      break;
    }
    
    // Scan the operands of this branch, replacing any uses of dst with
    // crit_mbb.
    for (unsigned i = 0, e = mii->getNumOperands(); i != e; ++i) {
      MachineOperand & mo = mii->getOperand(i);
      if (mo.isMachineBasicBlock() &&
          mo.getMachineBasicBlock() == dst) {
        found_branch = true;
        mo.setMachineBasicBlock(crit_mbb);
      }
    }
  }

  // TODO: This is tentative. It may be necessary to fix this code. Maybe
  // I am inserting too many gotos, but I am trusting that the asm printer
  // will optimize the unnecessary gotos.
  if(!found_branch) {
    TII->InsertBranch(*src, crit_mbb, (MachineBasicBlock*)0, emptyConditions);
  }

  /// Change all the phi functions in dst, so that the incoming block be
  /// crit_mbb, instead of src
  for(mii = dst->begin(); mii != dst->end(); mii++) {
    /// the first instructions are always phi functions.
    if(mii->getOpcode() != TargetInstrInfo::PHI)
      break;
    
    for (unsigned u = 0; u != mii->getNumOperands(); ++u)
      if (mii->getOperand(u).isMachineBasicBlock() &&
          mii->getOperand(u).getMachineBasicBlock() == src)
        mii->getOperand(u).setMachineBasicBlock(crit_mbb);
  }
}

bool BreakCriticalMachineEdges::runOnMachineFunction(MachineFunction& F) {
  std::vector<MachineBasicBlock *> SourceBlocks;
  std::vector<MachineBasicBlock *> DestBlocks;

  for(MachineFunction::iterator FI = F.begin(), FE = F.end(); FI != FE; ++FI) {
    for(MachineBasicBlock::succ_iterator SI = FI->succ_begin(),
        SE = FI->succ_end(); SI != SE; ++SI) {
      // predecessor with multiple successors, successor with multiple
      // predecessors.
      if (FI->succ_size() > 1 && (*SI)->pred_size() > 1) {
        SourceBlocks.push_back(FI);
        DestBlocks.push_back(*SI);
      }
    }
  }

  for(unsigned u = 0; u < SourceBlocks.size(); u++)
    splitCriticalEdge(SourceBlocks[u], DestBlocks[u]);

  return false;
}
